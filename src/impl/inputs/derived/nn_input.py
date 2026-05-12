"""
Neural-network-driven Input wrapper.

This derived Input loads a standard PyTorch checkpoint (.pth) together with a
metadata JSON (.meta.json) to reconstruct an nn.Module (MLP) and evaluates it as
u = f([x, t]) or u = f(x) depending on configuration.

Workflow:
 - Train an MLP with scripts/input/train_nn_input.py, which saves .pth + .meta.json
 - Construct NNInput with the path (base or .pth); time handling is inferred from metadata (time_invariant)
 - Bind to a System so the wrapper can finalize dimensions
"""

import json
import torch
from typing import Literal, Optional
from pathlib import Path


from src.core.inputs import Input
from src.core.systems import System
from src.utils.nn import MLP

__all__ = ["NNInput"]


class NNInput(Input):
    """Input backed by a reconstructed nn.Module.

    The model is expected to accept a tensor shaped [..., in_dim] and return
    [..., out_dim]. If metadata sets time_invariant=False, time is appended as
    the last feature to the state before forwarding to the model.
    """

    type: Literal['any', 'control', 'disturbance', 'uncertainty'] = 'any'
    system_class = System
    time_invariant: bool
    dim: int

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: Optional[str] = None,
    ) -> None:

        # Resolve paths
        p = Path(checkpoint_path)
        if p.is_dir():
            base = p / 'model'
            raw_path = base.with_suffix('.pth')
            meta_path = base.with_suffix('.meta.json')
        else:
            if p.suffix == '.pth':
                raw_path = p
                meta_path = p.with_suffix('.meta.json')
            else:
                raw_path = p.with_suffix('.pth')
                meta_path = p.with_suffix('.meta.json')

        # Load metadata and reconstruct module
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        sizes = meta['sizes']
        input_min = meta['input_min']
        input_max = meta['input_max']
        output_min = meta['output_min']
        output_max = meta['output_max']
        periodic_input_dims = meta.get('periodic_input_dims', [])
        meta_time_invariant = bool(meta.get('time_invariant', True))
        self.time_invariant = meta_time_invariant

        dev = torch.device(device) if device is not None else torch.device('cpu')
        self._module = MLP(
            sizes,
            input_min=input_min,
            input_max=input_max,
            output_min=output_min,
            output_max=output_max,
            periodic_input_dims=periodic_input_dims,
            device=dev,
        )
        sd = torch.load(raw_path, map_location=dev)
        state = sd['state_dict'] if isinstance(sd, dict) and 'state_dict' in sd else sd
        # Be tolerant to older checkpoints that didn't include registered buffers
        # (periodic_embedding.period, input/output normalizer stats). Our __init__ already
        # reconstructs these from metadata, so it's safe to ignore missing keys.
        self._module.load_state_dict(state, strict=False)
        self._module.eval()
        if device is not None:
            self._module.to(device)
        self._out_dim = int(sizes[-1])

    def to(self, device: str) -> "NNInput":
        self._module.to(device)
        return self

    def set_type(self, type: Literal['any', 'control', 'disturbance', 'uncertainty']) -> None:
        super().set_type(type)

    def bind(self, system: System) -> None:
        self._system = system
        self.dim = self._out_dim

    def input(self, state: torch.Tensor, time: float) -> torch.Tensor:
        if self._system is None:
            raise RuntimeError("NNInput must be bound to a System before calling input().")
        if not isinstance(state, torch.Tensor):
            x = torch.as_tensor(state)
        else:
            x = state
        orig_dtype = x.dtype
        # Move input to model's device/dtype
        model_device = next(self._module.parameters(), torch.tensor(0.)).device
        x = x.to(device=model_device)

        # Flatten batch dims
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        if not self.time_invariant:
            t = torch.full((x_flat.shape[0], 1), float(time), device=x_flat.device, dtype=x_flat.dtype)
            x_in = torch.cat([x_flat, t], dim=-1)
        else:
            x_in = x_flat

        with torch.no_grad():
            y = self._module(x_in)

        y = y.reshape(*orig_shape, self.dim).to(dtype=orig_dtype)
        return y

    def __repr__(self) -> str:
        mode = "x,t" if not self.time_invariant else "x"
        return f"NNInput(module={type(self._module).__name__}, mode={mode}, dim={getattr(self, 'dim', '?')})"
