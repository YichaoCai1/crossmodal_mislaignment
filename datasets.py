"""
Definition of datasets.
"""

import io
import json
import os
import numpy as np
import pandas as pd
import torch
from collections import Counter, OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize
from torchvision.datasets.folder import pil_loader


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order of elements encountered."""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Multimodal3DIdent(torch.utils.data.Dataset):
    """Multimodal3DIdent Dataset.

    Attributes:
        FACTORS (dict): names of factors for image and text modalities.
        DISCRETE_FACTORS (dict): names of discrete factors, respectively.
    """

    FACTORS = {
        "image": {
            0: "object_shape",
            1: "object_xpos",
            2: "object_ypos",
            # 3: "object_zpos",  # is constant
            4: "object_alpharot",
            5: "object_betarot",
            6: "object_gammarot",
            7: "spotlight_pos",
            8: "object_color",
            9: "spotlight_color",
            10: "background_color"
        },
        "text": {
            0: "object_shape",
            1: "object_xpos",
            2: "object_ypos",
            3: "spotlight_pos",
            4: "object_color_index",
            5: "splotlight_color_index",
            6: "background_color_index",
            7: "text_phrasing"
        }
    }
    
    SELECTIONS = [
        ["object_shape", "text_phrasing"],
        ["object_shape", "object_xpos", "text_phrasing"],
        ["object_shape", "object_xpos", "object_ypos", "text_phrasing"],
        ["object_shape", "object_xpos", "object_ypos", "spotlight_pos", "text_phrasing"],
        ["object_shape", "object_xpos", "object_ypos", "spotlight_pos", "object_color_index",
            "text_phrasing"],
        ["object_shape", "object_xpos", "object_ypos", "spotlight_pos", "object_color_index", 
            "splotlight_color_index", "text_phrasing"],
        ["object_shape", "object_xpos", "object_ypos", "spotlight_pos", "object_color_index", 
            "splotlight_color_index", "background_color_index", "text_phrasing"],
    ]

    DISCRETE_FACTORS = {
        "image": {
            0: "object_shape",
            1: "object_xpos",
            2: "object_ypos",
            # 3: "object_zpos",  # is constant
        },
        "text": {
            0: "object_shape",
            1: "object_xpos",
            2: "object_ypos",
            4: "object_color_index",
            5: "splotlight_color_index",
            6: "background_color_index",
            7: "text_phrasing"
        }
    }

    def __init__(self, data_dir, bias_type="selections", bias_id=6,  mode="train", transform=None,
            has_labels=True, vocab_filepath=None):
        """
        Args:
            data_dir (string): path to  directory.
            mode (string): name of data split, 'train', 'val', or 'test'.
            transform (callable): Optional transform to be applied.
            has_labels (bool): Indicates if the data has ground-truth labels.
            vocab_filepath (str): Optional path to a saved vocabulary. If None,
              the vocabulary will be (re-)created.
        """
        self.bias_type = bias_type
        self.bias_id = bias_id
        if bias_type != "selections":
            self.selected_semantics = self.SELECTIONS[-1]
        else:
            self.selected_semantics = self.SELECTIONS[bias_id]
        
        self.mode = mode
        self.transform = transform
        self.has_labels = has_labels
        self.data_dir = data_dir
        self.latents_text_filepath = \
            os.path.join(self.data_dir, f"latents_text_{self.bias_type}_{self.bias_id}.csv")
        self.latents_image_filepath = \
            os.path.join(self.data_dir, "latents_image.csv")
        self.text_filepath = \
            os.path.join(self.data_dir, "text", f"{self.bias_type}_{self.bias_id}", "text_raw.txt")
        self.image_dir = os.path.join(self.data_dir, "images")

        # load text
        text_in_sentences, text_in_words = self._load_text()
        self.text_in_sentences = text_in_sentences   # sentence-tokenized text
        self.text_in_words = text_in_words           # word-tokenized text

        # determine num_samples and max_sequence_length
        self.num_samples = len(self.text_in_sentences)
        self.max_sequence_length = \
            max([len(sent) for sent in self.text_in_words]) + 1  # +1 for "eos"
        
        # load or create the vocabulary (i.e., word <-> index maps)
        self.w2i, self.i2w = self._load_vocab(vocab_filepath)
        self.vocab_size = len(self.w2i)
        if vocab_filepath:
            self.vocab_filepath = vocab_filepath
        else:
            self.vocab_filepath = os.path.join(self.data_dir, f"vocab_{self.bias_type}_{self.bias_id}.json")

        # optionally, load ground-truth labels
        if has_labels:
            self.labels = self._load_labels()

        # create list of image filepaths
        image_paths = []
        width = int(np.ceil(np.log10(self.num_samples)))
        for i in range(self.num_samples):
            fp = os.path.join(self.image_dir, str(i).zfill(width) + ".png")
            image_paths.append(fp)
        self.image_paths = image_paths

    def get_w2i(self, word):
        try:
            return self.w2i[word]
        except KeyError:
            return "{unk}"  # special token for unknown words

    def _load_text(self):
        print(f"Tokenization of {self.mode} data...")

        # load raw text
        with open(self.text_filepath, "r") as f:
            text_raw = f.read()

        # create sentence-tokenized text
        text_in_sentences = sent_tokenize(text_raw)

        # create word-tokenized text
        text_in_words = [word_tokenize(sent) for sent in text_in_sentences]

        return text_in_sentences, text_in_words

    def _load_labels(self):

        # load image labels
        s_image = pd.read_csv(self.latents_image_filepath)

        # load text labels
        s_text = pd.read_csv(self.latents_text_filepath)

        # check if all factors are present
        for v in self.FACTORS["image"].values():
            assert v in s_image.keys()
        for v in self.selected_semantics:
            assert v in s_text.keys()

        # create label dict
        labels = {"s_image": s_image, "s_text": s_text}

        return labels

    def _create_vocab(self, vocab_filepath):
        print(f"Creating vocabulary as '{vocab_filepath}'...")

        if self.mode != "train":
            raise ValueError("Vocabulary should be created from training data")

        # initialize counter and word <-> index maps
        ordered_counter = OrderedCounter()  # counts occurrence of each word
        w2i = dict()  # word-to-index map
        i2w = dict()  # index-to-word map
        unique_words = []

        # add special tokens for padding, end-of-string, and unknown words
        special_tokens = ["{pad}", "{eos}", "{unk}"]
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for i, words in enumerate(self.text_in_words):
            ordered_counter.update(words)

        for w, _ in ordered_counter.items():
            if w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unique_words.append(w)
        if len(w2i) != len(i2w):
            raise ValueError("Mismatch between w2i and i2w mapping")

        # save vocabulary to disk
        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(vocab_filepath, "wb") as vocab_file:
            jd = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(jd.encode("utf8", "replace"))

        return vocab

    def _load_vocab(self, vocab_filepath=None):
        if vocab_filepath is not None:
            with open(vocab_filepath, "r") as vocab_file:
                vocab = json.load(vocab_file)
        else:
            new_filepath = os.path.join(self.data_dir, f"vocab_{self.bias_type}_{self.bias_id}.json")
            if os.path.exists(new_filepath):
                with open(new_filepath, "r") as vocab_file:
                    vocab = json.load(vocab_file)
            else:
                vocab = self._create_vocab(vocab_filepath=new_filepath)
        return (vocab["w2i"], vocab["i2w"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load image
        img_name = self.image_paths[idx]
        image = pil_loader(img_name)
        if self.transform is not None:
            image = self.transform(image)

        # load text
        words = self.text_in_words[idx]
        words = words + ["{eos}"]
        words = words + ["{pad}" for c in range(self.max_sequence_length-len(words))]
        indices = [self.get_w2i(word) for word in words]
        indices_onehot = torch.nn.functional.one_hot(
            torch.Tensor(indices).long(), self.vocab_size).float()

        # load labels
        if self.has_labels:
            s_image = {k: v[idx] for k, v in self.labels["s_image"].items()}
            s_text = {k: v[idx] for k, v in self.labels["s_text"].items()}
        else:
            s_image, s_text = None, None

        sample = {
            "image": image,
            "text": indices_onehot,
            "s_image": s_image,
            "s_text": s_text}
        return sample

    def __len__(self):
        return self.num_samples


class MultimodalMPI3DRealComplex(torch.utils.data.Dataset):
    """A base class for Multimodal dataset, considering captioning bias.
    """
    SEMANTICS = {
        "OBJ_COLOR": ["yellow color", "green color", "olive color", "red color"],
        "OBJ_SHAPE": ["coffee-cup", "tennis-ball", "croissant", " beer-cup"],
        "OBJ_SIZE": ["small size", "large size"],
        "CAMERA": ["top view", "center view", "bottom view"],
        "BACKGROUND": ["in a purple background", "in a sea-green background", "in a salmon background"],  
        "H_AXIS": [f"horizontal position {i}" for i in range(40)],
        "V_AXIS": [f"vertical position {ordinal}" for ordinal in ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
                    "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth",
                    "Twenty-first", "Twenty-second", "Twenty-third", "Twenty-fourth", "Twenty-fifth", "Twenty-sixth", "Twenty-seventh", "Twenty-eighth", "Twenty-ninth", "Thirtieth",
                    "Thirty-first", "Thirty-second", "Thirty-third", "Thirty-fourth", "Thirty-fifth", "Thirty-sixth", "Thirty-seventh", "Thirty-eighth", "Thirty-ninth", "Fortieth"
                ]]
    }
    
    SELECTION_BIAS = [
        ['OBJ_COLOR'],
        ['OBJ_COLOR', "OBJ_SHAPE"],
        ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE"],
        ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA"],
        ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA", "BACKGROUND"]
    ]   # For simplicity, we only consider a increasing order of selection here.
    
    
    PERTURBATION_BIAS = [
        [],
        ['BACKGROUND'],
        ['CAMERA', "BACKGROUND"],
        ["OBJ_SIZE", "CAMERA", "BACKGROUND"],
        ["OBJ_SHAPE", "OBJ_SIZE", "CAMERA", "BACKGROUND"],
    ]
    
    VAL_LATENTS=["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA", "BACKGROUND", "H_AXIS", "V_AXIS"]
    
    def __init__(self, data_dir, bias_type="selection", bias_id=4, mode="train", transform=None, vocab_filepath=None):
        self.bias_type = bias_type
        self.bias_id = bias_id
        
        if self.bias_type == "selection":
            self.selected_semantics = self.SELECTION_BIAS[bias_id]
            self.perturbed_semantics = []
        else:
            self.selected_semantics = self.SELECTION_BIAS[-1]
            self.perturbed_semantics = self.PERTURBATION_BIAS[bias_id]
        
        self.unbiased_semantics = list(set(self.selected_semantics) - set(self.perturbed_semantics))
        
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        self.data_dir_mode = os.path.join(data_dir, mode)
        self.latents_text_filepath = \
            os.path.join(self.data_dir_mode, f"text_semantics_{self.bias_type}_{self.bias_id}.csv")
        self.latents_image_filepath = \
            os.path.join(self.data_dir_mode, f"image_semantics.csv")
        self.text_filepath = \
            os.path.join(self.data_dir_mode, f"text_{self.bias_type}_{self.bias_id}.txt")
        
        mpi3d = \
            np.load(os.path.join(self.data_dir, "real3d_complicated_shapes_ordered.npz"))['images']
        self.data = mpi3d.reshape([4,4,2,3,3,40,40,64,64,3])
        self.latent_sizes = [4, 4, 2, 3, 3, 40, 40]
        
        # load text
        text_in_sentences, text_in_words = self._load_text()
        self.text_in_sentences = text_in_sentences
        self.text_in_words = text_in_words
        
        self.num_samples = len(self.text_in_sentences)
        self.max_sequence_length = \
            max([len(sent) for sent in self.text_in_words]) + 1  # +1 for "eos"
        
        self.w2i, self.i2w = self._load_vocab(vocab_filepath)
        self.vocab_size = len(self.w2i)
        if vocab_filepath:
            self.vocab_filepath = vocab_filepath
        else:
            self.vocab_filepath = os.path.join(self.data_dir, f"vocab_{self.bias_type}_{self.bias_id}.json")
        
        self.labels = self._load_labels()
        
    
    def get_w2i(self, word):
        try:
            return self.w2i[word]
        except KeyError:
            return "{unk}"  # special token for unknown words
        
    def _load_text(self):
        print(f"Tokenization of {self.mode} data...")

        # Load raw text and remove all periods
        with open(self.text_filepath, "r") as f:
            text_raw = f.read()

        # create sentence-tokenized text
        text_in_sentences = sent_tokenize(text_raw)

        # create word-tokenized text
        text_in_words = [word_tokenize(sent) for sent in text_in_sentences]

        return text_in_sentences, text_in_words


    def _load_labels(self):

        # load image labels
        s_image = pd.read_csv(self.latents_image_filepath)

        # load text labels
        s_text = pd.read_csv(self.latents_text_filepath)

        # check if all factors are present
        for v in self.VAL_LATENTS:
            assert v in s_image.keys()
        for v in self.selected_semantics:
            assert v in s_text.keys()

        # create label dict
        labels = {"s_image": s_image, "s_text": s_text}

        return labels

    def _create_vocab(self, vocab_filepath):
        print(f"Creating vocabulary as '{vocab_filepath}'...")

        if self.mode != "train":
            raise ValueError("Vocabulary should be created from training data")

        # initialize counter and word <-> index maps
        ordered_counter = OrderedCounter()  # counts occurrence of each word
        w2i = dict()  # word-to-index map
        i2w = dict()  # index-to-word map
        unique_words = []

        # add special tokens for padding, end-of-string, and unknown words
        special_tokens = ["{pad}", "{eos}", "{unk}"]
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for i, words in enumerate(self.text_in_words):
            ordered_counter.update(words)

        for w, _ in ordered_counter.items():
            if w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unique_words.append(w)
        if len(w2i) != len(i2w):
            print(unique_words)
            raise ValueError("Mismatch between w2i and i2w mapping")

        # save vocabulary to disk
        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(vocab_filepath, "wb") as vocab_file:
            jd = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(jd.encode("utf8", "replace"))

        return vocab
    

    def _load_vocab(self, vocab_filepath=None):
        if vocab_filepath is not None:
            with open(vocab_filepath, "r") as vocab_file:
                vocab = json.load(vocab_file)
        else:
            new_filepath = os.path.join(self.data_dir, f"vocab_{self.bias_type}_{self.bias_id}.json")
            if os.path.exists(new_filepath):
                with open(new_filepath, "r") as vocab_file:
                    vocab = json.load(vocab_file)
            else:
                vocab = self._create_vocab(vocab_filepath=new_filepath)
        return (vocab["w2i"], vocab["i2w"])
    
    
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # load image from latents
        img_latents = self.labels["s_image"].iloc[idx].tolist()
        
        img = self.data[tuple(img_latents)]     # advanced indexing -> (64, 64, 3)
        image = torch.from_numpy(img).float() / 255.0
        image = image.permute(2,0,1)    # (H, W, C) -> (C, H, W)
        
        if self.transform is not None:
            image = self.transform(image)
        
        # load text
        words = self.text_in_words[idx]
        words = words + ["{eos}"]
        words = words + ["{pad}" for c in range(self.max_sequence_length-len(words))]
        indices = [self.get_w2i(word) for word in words]
        indices_onehot = torch.nn.functional.one_hot(
            torch.Tensor(indices).long(), self.vocab_size).float()
        
        s_image = {k: v[idx] for k, v in self.labels["s_image"].items()}
        sample = {
            "image": image,
            "text": indices_onehot,
            "semantics": s_image        # Full semantics
        }
        
        return sample


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    dataset = MultimodalMPI3DRealComplex("./data/MPI3d_real_complex", bias_type="selection", bias_id=4, mode="train")

    # Select 10 random indices
    indices = np.random.choice(len(dataset), 10, replace=False)

    # Retrieve images
    images = [dataset[i]["image"] for i in indices]

    # Create a grid of images in one row
    image_grid = make_grid(images, nrow=10, padding=2, normalize=True)

    # Plot images
    plt.figure(figsize=(20, 5))
    plt.imshow(image_grid.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
    plt.axis("off")
    plt.savefig("mpi_sample.pdf", format="pdf", dpi=600, bbox_inches="tight")
    plt.title("10 Random Images from MultimodalMPI3DRealComplex Dataset")
    plt.show()
    
    # # Get the third image (index 2)
    # third_image = dataset[2]["image"]  # This should be a tensor of shape (C, H, W)

    # # Convert tensor to NumPy array for plotting
    # img_np = third_image.permute(1, 2, 0).numpy()  # Convert to (H, W, C)

    # # Plot and save
    # plt.figure(figsize=(5, 5))
    # plt.imshow(img_np)
    # plt.axis("off")
    # plt.title("Third Image from Dataset")
    # plt.savefig("third_image.png", format="png", dpi=600, bbox_inches="tight")
    # plt.show()