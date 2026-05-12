"""Neural network building blocks: normalization layers and MLP with periodic embeddings."""
from __future__ import annotations

import torch
from torch import nn, Tensor

__all__ = ["Normalize", "Unnormalize", "PeriodicEmbedding", "MLP"]


class Normalize(nn.Module):
    def __init__(self, mean, halfwidth):
        super().__init__()
        # Register as buffers so nn.Module.to(device) moves them automatically
        self.register_buffer('mean', mean)
        self.register_buffer('halfwidth', halfwidth)

    def forward(self, tensor: Tensor) -> Tensor:
        # Assumes buffers are already on the same device as the module
        return tensor.sub(self.mean).div(self.halfwidth)
    
    def __repr__(self):
        return self.__class__.__name__ + \
            '(mean={0}, halfwidth={1})'.format(self.mean, self.halfwidth)
    
    
class Unnormalize(nn.Module):
    def __init__(self, mean, halfwidth):
        super().__init__()
        # Register as buffers so nn.Module.to(device) moves them automatically
        self.register_buffer('mean', mean)
        self.register_buffer('halfwidth', halfwidth)

    def forward(self, tensor: Tensor) -> Tensor:
        # Assumes buffers are already on the same device as the module
        return tensor.mul(self.halfwidth).add(self.mean)
    
    def __repr__(self):
        return self.__class__.__name__ + \
            '(mean={0}, halfwidth={1})'.format(self.mean, self.halfwidth)
    
class PeriodicEmbedding(nn.Module):
    def __init__(self, period, periodic_dims):
        super().__init__()
        # Register as buffer so nn.Module.to(device) moves it automatically
        self.register_buffer('period', period)
        self.periodic_dims = periodic_dims
    def forward(self, x):
        nonperiodic_dims = [i for i in range(x.shape[-1]) 
                            if i not in self.periodic_dims]
        # periodic embedding
        periodic_x = 2*torch.pi*x[:, self.periodic_dims].div(
            self.period[self.periodic_dims])
        return torch.concatenate((
            x[:, nonperiodic_dims], 
            torch.sin(periodic_x), 
            torch.cos(periodic_x)), dim=-1)

class MLP(nn.Module):
    def __init__(self, sizes, input_min, input_max, 
                 output_min, output_max, periodic_input_dims, device):
        """sizes: list of vector sizes; x -> y -> z"""
        super().__init__()
        def mean_halfwidth(min, max):
            min = torch.tensor(min, device=device)
            max = torch.tensor(max, device=device)
            return (min+max)/2, (max-min)/2
        input_mean, input_halfwidth = mean_halfwidth(input_min, input_max)
        output_mean, output_halfwidth = mean_halfwidth(output_min, output_max)
        # periodic embedding layer
        self.periodic_embedding = PeriodicEmbedding(
            2*input_halfwidth, periodic_input_dims)
        # input normalization layer
        nonperiodic_input_dims = [i for i in range(len(input_min)) 
                                  if i not in periodic_input_dims]
        input_mean = torch.concatenate((
            input_mean[nonperiodic_input_dims], 
            torch.zeros(2*len(periodic_input_dims), device=device)), dim=0)
        input_halfwidth = torch.concatenate((
            input_halfwidth[nonperiodic_input_dims], 
            torch.ones(2*len(periodic_input_dims), device=device)), dim=0)
        self.input_normalizer = Normalize(input_mean, input_halfwidth)
        # network layers
        layers = []
        input_size = sizes[0] + len(periodic_input_dims)
        for size in sizes[1:]:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        layers = layers[:-1] # remove last ReLU
        self.nn = nn.Sequential(*layers)
        # output unnormalization layer
        self.output_unnormalizer = Unnormalize(output_mean, output_halfwidth)
    def forward(self, x):
        return self.output_unnormalizer(self.nn(
            self.input_normalizer(self.periodic_embedding(x))))