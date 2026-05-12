import torch

from src.core.inputs import Input
from src.core.systems import System

__all__ = ["UniformRandomInput"]

class UniformRandomInput(Input):
    type = 'any'
    system_class = System
    dim: int
    system: System
    time_invariant = False

    def input(self, state, time):
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.as_tensor(state)
        else:
            state_tensor = state
        if self.type == 'any':
            raise ValueError('Input type must be set before calling input()')
        if self.type == 'control':
            low, high = self.system.control_limits(state_tensor, time)
        elif self.type == 'disturbance':
            low, high = self.system.disturbance_limits(state_tensor, time)
        elif self.type == 'uncertainty':
            low, high = self.system.uncertainty_limits(state_tensor, time)
        noise = torch.rand(
            *state_tensor.shape[:-1],
            self.dim,
            dtype=state_tensor.dtype,
            device=state_tensor.device,
        )
        return noise * (high - low) + low
    
    def bind(self, system):
        self.system = system
        if self.type == 'any':
            raise ValueError('Input type must be set before binding to a system')
        if self.type == 'control':
            self.dim = system.control_dim
        elif self.type == 'disturbance':
            self.dim = system.disturbance_dim
        elif self.type == 'uncertainty':
            self.dim = system.state_dim
