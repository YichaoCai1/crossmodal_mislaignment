'''
 # @ Author: Yichao Cai
 # @ Create Time: 2025-02-28 01:46:56
 # @ Description: Datasets: MPI3DReal, Causal3DIdent
 '''
 
import io
import os
import json
import random
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt

class MultimodalMPI3DRealComplex(torch.utils.data.Dataset):
    """A base class for Multimodal dataset, considering captioning bias.
    """
    SEMANTICS = {
        "OBJ_COLOR": ["yellow", "green", "olive", "red"],
        "OBJ_SHAPE": ["coffee-cup", "tennis-ball", "croissant", " beer-cup"],
        "OBJ_SIZE": ["small size", "large size"],
        "CAMERA": ["top view", "center view", "bottom view"],
        "BACKGROUND": ["in a purple background", "in a sea-green background", "in a salmon background"]
    }
    
    SELECTION_BIAS = [
        ['OBJ_COLOR'],
        ['OBJ_COLOR', "OBJ_SHAPE"],
        ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE"],
        ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA"],
        ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA", "BACKGROUND"],
    ]   # For simplicity, we only consider a increasing order of selection here.
    
    # PERTURBATION_BIAS = [
    #     [],
    #     ['OBJ_COLOR'],
    #     ['OBJ_COLOR', "OBJ_SHAPE"],
    #     ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE"],
    #     ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA"],
    # ]
    
    PERTURBATION_BIAS = [
        [],
        ['BACKGROUND'],
        ['CAMERA', "BACKGROUND"],
        ["OBJ_SIZE", "CAMERA", "BACKGROUND"],
        ["OBJ_SHAPE", "OBJ_SIZE", "CAMERA", "BACKGROUND"],
    ]
    
    def __init__(self, train_ratio=0.8):
        super().__init__()
        self.train_ratio=train_ratio
        
        self.latent_sizes = [4, 4, 2, 3, 3, 40, 40]
        total_size = np.prod(self.latent_sizes)
        self.indices = list(range(total_size))
        
        self.origin_semantic_order = ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA",
                                "BACKGROUND", "H_AXIS", "V_AXIS"]
    
    def _generate_descrition_and_latents(self, latents, selection=4, perturb=0):
        selected_semantics = self.SELECTION_BIAS[selection]
        perturbable_semantics = self.PERTURBATION_BIAS[perturb]

        semantics = {}
        text_latents = []
        for id, k in enumerate(self.origin_semantic_order):
            if k in selected_semantics:
                text_latent = latents[id]
                if (k in perturbable_semantics) and random.random() <= 0.9:
                    text_latent = random.choice(range(len(self.SEMANTICS[k])))
                semantics[k] = self.SEMANTICS[k][text_latent]
                text_latents.append(text_latent)

        # Multiple templates for each selection level
        templates_dict = {
            0: [
                "An object colored {OBJ_COLOR}.",
                "It has a {OBJ_COLOR} appearance.",
                "Something with {OBJ_COLOR}."
            ],
            1: [
                "A {OBJ_SHAPE} that is {OBJ_COLOR}.",
                "The {OBJ_COLOR} {OBJ_SHAPE}.",
                "An object shaped like a {OBJ_SHAPE}, colored {OBJ_COLOR}."
            ],
            2: [
                "A {OBJ_SIZE} {OBJ_SHAPE} in {OBJ_COLOR}.",
                "{OBJ_COLOR}, {OBJ_SIZE}, {OBJ_SHAPE}.",
                "The object is {OBJ_SIZE}, shaped as a {OBJ_SHAPE}, and colored {OBJ_COLOR}."
            ],
            3: [
                "A {OBJ_SIZE} {OBJ_SHAPE} in {OBJ_COLOR}, seen from {CAMERA}.",
                "Viewed from {CAMERA}, a {OBJ_COLOR}, {OBJ_SIZE} {OBJ_SHAPE}.",
                "A {OBJ_SIZE} {OBJ_SHAPE} with {OBJ_COLOR}, perspective: {CAMERA}."
            ],
            4: [
                "A {OBJ_SIZE} {OBJ_SHAPE} in {OBJ_COLOR}, viewed from {CAMERA}, {BACKGROUND}.",
                "From {CAMERA}, you see a {OBJ_COLOR}, {OBJ_SIZE} {OBJ_SHAPE}, {BACKGROUND}.",
                "A {OBJ_SIZE} {OBJ_SHAPE}, {OBJ_COLOR}, placed {BACKGROUND}, observed from {CAMERA}."
            ]
        }

        templates = templates_dict.get(selection, [", ".join([semantics[k] for k in selected_semantics]) + "."])

        template = random.choice(templates)
        text = template.format(**semantics)

        return text, text_latents

    def __len__(self):
        return len(self.indices)


    def __getitem__(self, index):
        original_index = self.indices[index]
        latents = np.unravel_index(original_index, self.latent_sizes)   # image semantics
        
        text_selections, text_latents_selections = [], [] 
        text_perturbations, text_latents_perturbations = [], [] 
        for i in range(5):
            text, text_latents = self._generate_descrition_and_latents(latents, selection=i)
            text_selections.append(text)
            text_latents_selections.append(text_latents)

            text, text_latents = self._generate_descrition_and_latents(latents, perturb=i)
            text_perturbations.append(text)
            text_latents_perturbations.append(text_latents)
            
        return latents, text_selections, text_latents_selections, text_perturbations, text_latents_perturbations


    def generate_files(self, output_dir, seed=42):
        np.random.seed(seed)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Shuffle indices for random splitting
        indices = np.arange(len(self.indices))
        np.random.shuffle(indices)
        split_idx = int(self.train_ratio * len(indices))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        # Initialize storage
        train_latents, test_latents = [], []
        train_text_latents_selection = [[] for _ in range(5)]
        test_text_latents_selection = [[] for _ in range(5)]
        train_text_latents_perturbation = [[] for _ in range(5)]
        test_text_latents_perturbation = [[] for _ in range(5)]
        train_texts_selection = ["" for _ in range(5)]
        test_texts_selection = ["" for _ in range(5)]
        train_texts_perturbation = ["" for _ in range(5)]
        test_texts_perturbation = ["" for _ in range(5)]

        for id in tqdm(train_indices):
            latents, text_selections, text_latents_selections, text_perturbations, text_latents_perturbations = self.__getitem__(id)
            train_latents.append(latents)
            for i in range(5):
                train_text_latents_selection[i].append(text_latents_selections[i])
                train_text_latents_perturbation[i].append(text_latents_perturbations[i])
                train_texts_selection[i] += text_selections[i] + "\n"
                train_texts_perturbation[i] += text_perturbations[i] + "\n"

        for id in tqdm(test_indices):
            latents, text_selections, text_latents_selections, text_perturbations, text_latents_perturbations = self.__getitem__(id)
            test_latents.append(latents)
            for i in range(5):
                test_text_latents_selection[i].append(text_latents_selections[i])
                test_text_latents_perturbation[i].append(text_latents_perturbations[i])
                test_texts_selection[i] += text_selections[i] + "\n"
                test_texts_perturbation[i] += text_perturbations[i] + "\n"

        # # Save latent data
        pd.DataFrame(train_latents, columns=self.origin_semantic_order).to_csv(os.path.join(train_dir, "image_semantics.csv"), index=False)
        pd.DataFrame(test_latents, columns=self.origin_semantic_order).to_csv(os.path.join(test_dir, "image_semantics.csv"), index=False)

        # Save text latents and text files
        for i in range(5):
            pd.DataFrame(train_text_latents_selection[i], columns=self.SELECTION_BIAS[i]).to_csv(os.path.join(train_dir, f"text_semantics_selection_{i}.csv"), index=False)
            pd.DataFrame(test_text_latents_selection[i], columns=self.SELECTION_BIAS[i]).to_csv(os.path.join(test_dir, f"text_semantics_selection_{i}.csv"), index=False)
            
            pd.DataFrame(train_text_latents_perturbation[i], columns=self.SELECTION_BIAS[-1]).to_csv(os.path.join(train_dir, f"text_semantics_perturbation_{i}.csv"), index=False)
            pd.DataFrame(test_text_latents_perturbation[i], columns=self.SELECTION_BIAS[-1]).to_csv(os.path.join(test_dir, f"text_semantics_perturbation_{i}.csv"), index=False)

            with open(os.path.join(train_dir, f"text_selection_{i}.txt"), 'w') as wf:
                wf.write(train_texts_selection[i])
            with open(os.path.join(test_dir, f"text_selection_{i}.txt"), 'w') as wf:
                wf.write(test_texts_selection[i])

            with open(os.path.join(train_dir, f"text_perturbation_{i}.txt"), 'w') as wf:
                wf.write(train_texts_perturbation[i])
            with open(os.path.join(test_dir, f"text_perturbation_{i}.txt"), 'w') as wf:
                wf.write(test_texts_perturbation[i])

        print("Done Done London!")


if __name__ == "__main__":
    dataset = MultimodalMPI3DRealComplex(train_ratio=0.9)
    print("Dataset Initialized...\n")
    
    dataset.generate_files("./MPI3d_real_complex")
