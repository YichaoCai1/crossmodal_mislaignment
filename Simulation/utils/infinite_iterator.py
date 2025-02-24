"""
This code originates from the following projects:
- https://github.com/brendel-group/cl-ica
- https://github.com/ysharma1126/ssl_identifiability
"""

import torch
from typing import Iterable
from itertools import combinations

class InfiniteIterator:
    """Infinitely repeat the iterable."""
    def __init__(self, iterable: Iterable):
        self._iterable = iterable
        self.iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(2):
            try:
                return next(self.iterator)
            except StopIteration:
                # reset iterator
                del self.iterator
                self.iterator = iter(self._iterable)


class PowersetIndexer:
    def __init__(self, dim: int):
        """
        Initialize the powerset indexer with a specified dimension (number of elements).
        
        :param dim: The number of elements in the original set.
        """
        self.dim = dim
        self.index_map = list(self._generate_index_map())
        self.reverse_map = {tuple(indices): i for i, indices in enumerate(self.index_map)}
    
    def _generate_index_map(self):
        """
        Generate an ordered mapping of indices representing the powerset.
        """
        for r in range(1, self.dim + 1):
            yield from combinations(range(self.dim), r)
    
    def __getitem__(self, index):
        """
        Retrieve the indices subset corresponding to the given index.
        
        :param index: The index in the powerset list.
        :return: A list containing the corresponding subset of indices.
        """
        if 0 <= index < len(self.index_map):
            return self.index_map[index]
        else:
            raise IndexError("Index out of range.")

    def __len__(self):
        """
        Get the total number of subsets.
        """
        return len(self.index_map)


if __name__ == '__main__':
    
    full_tensor = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # Example values of one-dim tensor
    # full_tensor = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]*3)  # Example values of batched tensor

    dim = len(full_tensor if len(full_tensor.shape)==1 else full_tensor[-1])
    theta = PowersetIndexer(dim)

    indices_list = [0, 10, 55, 175, 385, 637, 847, 967, 1012, 1022]
    for i in indices_list:
        subset_indices = theta[i]
        subset_values = full_tensor[...,subset_indices]
        print(f"Subset indices: {subset_indices}, \t Subset values: {subset_values}")  
    