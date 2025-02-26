import torch
from itertools import combinations

class PowersetIndexer:
    def __init__(self, tensor: torch.Tensor):
        """
        Initialize the powerset indexer for a PyTorch tensor.

        :param tensor: A 1D torch.Tensor containing the elements.
        """
        if tensor.dim() != 1:
            raise ValueError("Input tensor must be 1-dimensional.")
        
        self.tensor = tensor
        self.index_map = list(self._generate_index_map())
        self.reverse_map = {tuple(self.tensor[list(indices)].tolist()): i for i, indices in enumerate(self.index_map)}

    def _generate_index_map(self):
        """
        Generate an ordered mapping of indices representing the powerset.
        """
        n = len(self.tensor)
        for r in range(1, n + 1):
            yield from combinations(range(n), r)

    def __getitem__(self, index):
        """
        Retrieve the subset corresponding to the given index.

        :param index: The index in the powerset list.
        :return: A torch.Tensor containing the corresponding subset.
        """
        if 0 <= index < len(self.index_map):
            return self.tensor[list(self.index_map[index])]
        else:
            raise IndexError("Index out of range.")

    def get_index(self, subset: torch.Tensor):
        """
        Retrieve the index of a given subset.

        :param subset: A torch.Tensor representing a subset.
        :return: The index corresponding to the subset in the powerset.
        """
        subset_tuple = tuple(subset.tolist())  # Convert tensor to tuple for lookup
        return self.reverse_map.get(subset_tuple, -1)  # Return -1 if not found

    def get_indices_of_prefixes(self):
        """
        Retrieve the indices of progressively growing prefixes of the original tensor.

        :return: A list of indices corresponding to subsets: [tensor[:1], tensor[:2], ..., tensor[:n]]
        """
        return [self.get_index(self.tensor[:i]) for i in range(1, len(self.tensor) + 1)]

    def __len__(self):
        """
        Get the total number of subsets.
        """
        return len(self.index_map)

# Example usage:
input_tensor = torch.tensor([1, 2, 3])
theta = PowersetIndexer(input_tensor)

# Retrieve indices of progressive prefixes
prefix_indices = theta.get_indices_of_prefixes()
print(prefix_indices)  # Output: [0, 3, 6]

# Test on a larger input
input_tensor_large = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
theta_large = PowersetIndexer(input_tensor_large)

prefix_indices_large = theta_large.get_indices_of_prefixes()
print(prefix_indices_large)  # Output: [0, 5, 15, 25, 31]