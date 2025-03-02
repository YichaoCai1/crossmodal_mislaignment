"""
Definition of latent spaces for the multimodal setup.

Parts of this code originate from the files spaces.py and latent_spaces.py
from the following projects:
- https://github.com/brendel-group/cl-ica
- https://github.com/ysharma1126/ssl_identifiability
"""

from typing import Callable, List
from abc import ABC, abstractmethod
import numpy as np
import torch


class Space(ABC):
    @abstractmethod
    def uniform(self, size, device):
        pass

    @abstractmethod
    def normal(self, mean, std, size, device):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass
    
    @property
    @abstractmethod
    def perturb_subset(self):
        pass
    
    @property
    @abstractmethod
    def select_subset(self):
        pass
    
    @property
    @abstractmethod
    def inv_subset(self):
        pass

class NRealSpace(Space):
    def __init__(self, n, selected_indices=None, perturb_indices=[]):
        self.n = n
        if selected_indices is None:    # Default selecting all semantics
            selected_indices = list(range(n))
        else:
            for id in selected_indices:
                assert (id>=0) and (id<n)
        self.selected_indices = selected_indices
        
        self.perturb_mask = [0] * n        # 0 for non-perturbable
        if len(perturb_indices) != 0:
            for id in perturb_indices:
                assert (id>=0) and (id<n)
                self.perturb_mask[id] = 1
        
        self.perturb_indices = sorted(list(set(selected_indices) & set(perturb_indices)))
        self.inv_indices = sorted(list(set(selected_indices) - set(self.perturb_indices)))
    
    @property
    def dim(self):
        return self.n
    
    @property
    def perturb_subset(self):
        return self.perturb_indices
    
    @property
    def select_subset(self):
        return self.selected_indices
    
    @property
    def inv_subset(self):
        return self.inv_indices

    def uniform(self, size, device="cpu"):
        raise NotImplementedError("Not defined on R^n")

    def normal(self, mean, std, size, device="cpu", change_prob=1., Sigma=None, shift_ids=None):
        """Sample from a Normal distribution in R^N.
        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """
        
        change_mask = torch.FloatTensor(self.perturb_mask).to(device)
        
        if mean is None:
            mean = torch.zeros(self.n)
            change_mask = torch.FloatTensor([1]*self.n).to(device)  # for marginal sampling

        if len(mean.shape) == 1 and mean.shape[0] == self.n:
            mean = mean.unsqueeze(0)
        if not torch.is_tensor(std):
            std = torch.ones(self.n) * std
        if len(std.shape) == 1 and std.shape[0] == self.n:
            std = std.unsqueeze(0)
        assert len(mean.shape) == 2
        assert len(std.shape) == 2

        if torch.is_tensor(mean):
            mean = mean.to(device)
        if torch.is_tensor(std):
            std = std.to(device)    
        
        # returning bianry output governed by the changing_probability
        change_indices = torch.distributions.binomial.Binomial(probs=change_prob).sample((size, self.n)).to(device)     
        if Sigma is not None:
            changes = np.random.multivariate_normal(np.zeros(self.n), Sigma, size)
            changes = torch.FloatTensor(changes).to(device)     # changing with given in-block dipendencies
        else:
            changes = torch.randn((size, self.n), device=device) * std      # changing with std = 1
        
        # Apply deterministic skewed heavy-tailed transformation to selected indices
        if shift_ids is not None:
            for idx in shift_ids:
                change_indices[...,idx] = 1.
                change_mask[...,idx] = 1.
                changes[...,idx] = torch.sign(changes[...,idx]) * (torch.abs(changes[...,idx]) ** 1.5)
        
        return mean + change_mask * change_indices * changes  


class LatentSpace:
    """Combines a topological space with a marginal and conditional density to sample from."""

    def __init__(
        self, space: Space, sample_marginal: Callable, sample_conditional: Callable
    ):
        self.space = space
        self._sample_marginal = sample_marginal
        self._sample_conditional = sample_conditional

    @property
    def sample_conditional(self):
        if self._sample_conditional is None:
            raise RuntimeError("sample_conditional was not set")
        return lambda *args, **kwargs: self._sample_conditional(
            self.space, *args, **kwargs
        )

    @sample_conditional.setter
    def sample_conditional(self, value: Callable):
        assert callable(value)
        self._sample_conditional = value

    @property
    def sample_marginal(self):
        if self._sample_marginal is None:
            raise RuntimeError("sample_marginal was not set")
        return lambda *args, **kwargs: self._sample_marginal(
            self.space, *args, **kwargs
        )

    @sample_marginal.setter
    def sample_marginal(self, value: Callable):
        assert callable(value)
        self._sample_marginal = value

    @property
    def dim(self):
        return self.space.dim

    @property
    def perturb_subset(self):
        return self.space.perturb_subset
    
    @property
    def select_subset(self):
        return self.space.select_subset
    
    @property
    def inv_subset(self):
        return self.space.inv_subset
    

class ProductLatentSpace(LatentSpace):
    """A latent space which is the cartesian product of other latent spaces."""

    def __init__(self, spaces: List[LatentSpace]):
        """Assumes that the list of spaces is [s, m_x, m_t]."""
        self.spaces = spaces

        # determine dimensions, assuming the ordering [s, m_x, m_t]
        assert len(spaces) in (1, 3)  # either [s] or [s, m_x, m_t]
        self.semantics_n = spaces[0].dim
        self.perturb_indices = spaces[0].perturb_subset
        self.select_indices = spaces[0].select_subset
        self.inv_indices = spaces[0].inv_subset
        self.rep_n = len(self.select_indices) - len(self.perturb_indices)
        assert self.rep_n > 0
        
        self.modality_n = 0
        if len(spaces) > 1:
            assert spaces[1].dim == spaces[2].dim  # can be relaxed
            self.modality_n = spaces[1].dim

    def sample_conditional(self, z, size, **kwargs):
        z_new = []
        n = 0
        for i, s in enumerate(self.spaces):
            if len(z.shape) == 1:
                z_s = z[n : n + s.space.n]
            else:
                z_s = z[:, n : n + s.space.n]
            
            n += s.space.n
            z_new.append(s.sample_conditional(z=z_s, size=size, **kwargs))

        return torch.cat(z_new, -1)

    def sample_marginal(self, size, **kwargs):
        z = [s.sample_marginal(size=size, **kwargs) for s in self.spaces]
        return torch.cat(z, -1)

    def sample_zx_zt(self, size, device):
        z = self.sample_marginal(size=size, device=device) 
        z_tilde = self.sample_conditional(z, size=size, device=device)  
        
        # decompose
        s, mx, _ = self.z_to_sm(z)
        zx = torch.cat((s, mx), dim=-1)
        
        s_tilde, _, mt = self.z_to_sm(z_tilde)
        s_theta_tilde = s_tilde[..., self.select_indices]    # semantics in text
        zt = torch.cat((s_theta_tilde, mt), dim=-1)
        
        return zx, zt, s, s_theta_tilde, mx, mt

    def z_to_sm(self, z):
        ns, nm = self.semantics_n, self.modality_n
        ix_s = torch.tensor(range(0, ns), dtype=int)
        ix_mx = torch.tensor(range(ns, ns + nm), dtype=int)
        ix_mt = torch.tensor(range(ns + nm, ns + nm*2), dtype=int)
        s = z[:, ix_s]
        mx = z[:, ix_mx]
        mt = z[:, ix_mt]
        return s, mx, mt

    @property
    def dim(self):
        dim_zx = self.semantics_n + self.modality_n
        dim_zt = len(self.select_indices) + self.modality_n
        dim_rep = self.rep_n
        return dim_zx, dim_zt, dim_rep