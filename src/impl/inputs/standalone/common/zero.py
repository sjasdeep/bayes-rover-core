import torch

from src.core.inputs import Input
from src.core.systems import System

__all__ = ["ZeroInput"]

class ZeroInput(Input):
    type = 'any'
    system_class = System
    dim: int
    time_invariant = True

    def input(self, state, time):
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.as_tensor(state)
        else:
            state_tensor = state
        return torch.zeros(
            *state_tensor.shape[:-1],
            self.dim,
            dtype=state_tensor.dtype,
            device=state_tensor.device,
        )
    
    def bind(self, system):
        if self.type == 'any':
            raise ValueError('Input type must be set before binding to a system')
        if self.type == 'control':
            self.dim = system.control_dim
        elif self.type == 'disturbance':
            self.dim = system.disturbance_dim
        elif self.type == 'uncertainty':
            self.dim = system.state_dim
