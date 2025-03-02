'''
 # @ Author: Yichao Cai
 # @ Create Time: 2025-02-28 01:46:56
 # @ Description: Datasets: MPI3DReal, Causal3DIdent
 '''
 
import io
import os
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize
import torch.utils
import torch.utils.data
from torchvision.datasets.folder import pil_loader


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order of elements encountered."""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
    
    
class MultimodalMPI3DReal(torch.utils.data.Dataset):
    """A base class for Multimodal dataset, considering captioning bias.
    """
    SEMANTICS = {
        "OBJ_COLOR": ["white", "green", "red", "blue", "brown", "olive"],
        "OBJ_SHAPE": ["cone", "cube", "cylinder", "hexagonal", "pyramid", "sphere"],
        "OBJ_SIZE": ["small", "large"],
        "CAMERA": ["top view", "center view", "bottom view"],
        "BACKGROUND": ["purple", "sea green", "salmon"],
        "H_AXIS": ["at the extreme left edge", "far left", "clearly positioned left", "left of center", "just slightly left of center",
                    "exactly at the center-left", "leaning toward the center", "just before the middle", "precisely at the center",
                    "just beyond the middle", "leaning toward the right", "exactly at the center-right", "just slightly right of center",
                    "right of center", "clearly positioned right", "far right", "at the extreme right edge",
                    "toward the right boundary", "very close to the right edge", "near the rightmost limit"],
        "V_AXIS": [    "at the extreme bottom", "very low in the frame", "close to the lower boundary", "low but not at the bottom",
                    "just below center", "approaching the vertical center", "slightly below the center", "precisely at the middle",
                    "slightly above the center", "near the vertical midpoint", "just above center", "approaching the upper half",
                    "clearly positioned high", "near the top but not at the highest point", "high but still below the top edge",
                    "very high in the frame", "close to the upper boundary", "almost at the top limit", "near the topmost edge",
                    "at the extreme top edge"]
    }   
    
    TEMPLATES = [ "A {size} {shape}, {color}, is positioned {horizontal}, {vertical}. Captured from a {camera}, against a {background} background.",
            "Against a {background} background, a {size} {shape} appears {horizontal} and {vertical}, seen from a {camera} perspective. The object is {color}.",
            "Positioned {horizontal} and {vertical}, a {size} {shape} in {color} is viewed from a {camera} perspective with a {background} background.",
    ]   # Denoting the text-specific noises, i.e., m_t.
    
    
    def __init__(self, data_dir, mode="train", 
                 perturb_semantics=[], select_semantics=[],
                 transform=None, has_labels=True, vocab_filepath=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.has_labels = has_labels
        self.data_dir = data_dir
        self.data_dir_mode = os.path.join(data_dir, mode)
        