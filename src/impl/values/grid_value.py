"""
Query-only value function reconstructed from cached HJ reachability output.

The class assumes caches follow the conventions produced by the build scripts:
  - metadata['grid_coordinate_vectors'] provides per-dimension axis arrays
  - values are stored as [T, *grid_shape]
  - gradients are stored as [T, *grid_shape, state_dim]
Everything is kept on CPU for predictable behaviour; callers may move tensors
to other devices after querying.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from ...core.hj_reachability import HJReachabilityDynamics
from ...core.systems import System
from ...core.values import Value
from ...utils.registry import get_system_class
from ...utils.grids import (
    compute_strides as _compute_strides_util,
    exact_axis_indices as _exact_axis_indices_util,
    exact_state_indices as _exact_state_indices_util,
    flatten_multi_index as _flatten_multi_index_util,
    nearest_state_indices as _nearest_state_indices_util,
    nearest_time_index as _nearest_time_index_util,
)

__all__ = ["GridValue"]


class GridValue(Value):
    """Value lookup backed by dense grid caches produced offline."""

    def __init__(
        self,
        values,
        times,
        gradients,
        metadata: Dict,
        interpolate: bool,
        hj_dynamics: HJReachabilityDynamics,
    ) -> None:
        coord_vecs = metadata.get("grid_coordinate_vectors")
        if not coord_vecs:
            raise ValueError("metadata['grid_coordinate_vectors'] is required.")

        self.metadata = metadata
        self.interpolate = bool(interpolate)
        self.hj_dynamics = hj_dynamics

        self._axes: List[Tensor] = [
            torch.as_tensor(np.asarray(axis), dtype=torch.float32) for axis in coord_vecs
        ]
        self.state_dim = len(self._axes)
        self.grid_shape = tuple(int(axis.numel()) for axis in self._axes)

        self._times = (
            times.detach().cpu().to(torch.float32)
            if isinstance(times, torch.Tensor)
            else torch.as_tensor(np.asarray(times), dtype=torch.float32)
        )
        self.num_times = int(self._times.numel())

        vals = (
            values.detach().cpu().to(torch.float32)
            if isinstance(values, torch.Tensor)
            else torch.as_tensor(np.asarray(values), dtype=torch.float32)
        )
        grads = (
            gradients.detach().cpu().to(torch.float32)
            if isinstance(gradients, torch.Tensor)
            else torch.as_tensor(np.asarray(gradients), dtype=torch.float32)
        )
        self._values = vals.movedim(0, -1)  # [*grid, T]
        self._grads = grads.movedim(0, -2)  # [*grid, T, D]

        self._state_strides = _compute_strides_util(list(self.grid_shape))

        bindings = metadata.get("bindings", {}) or {}
        self.control_given = (bindings.get("control") or {}).get("kind") == "input"
        self.disturbance_given = (bindings.get("disturbance") or {}).get("kind") == "input"
        self.uncertainty_given = (bindings.get("uncertainty") or {}).get("kind") == "input"

        self._system_cache: System | None = None

    @property
    def times(self) -> np.ndarray:
        """Return the cached time axis as a NumPy array."""
        return self._times.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Core queries
    # ------------------------------------------------------------------ #
    def value(self, state: Tensor, time: float | Tensor, *, interpolate: bool | None = None) -> Tensor:
        interpolate = self.interpolate if interpolate is None else interpolate
        state_cpu, orig_shape, orig_dtype, orig_device = self._prepare_state(state)
        time_cpu = self._prepare_time(time, state_cpu.shape[0])
        idx_state, idx_time = self._index(state_cpu, time_cpu, interpolate)

        flat_idx = _flatten_multi_index_util(idx_state, self._state_strides)
        vals_flat = self._values.reshape(-1, self.num_times)
        out = vals_flat[flat_idx, idx_time].reshape(orig_shape)
        return out.to(device=orig_device, dtype=orig_dtype)

    def gradient(self, state: Tensor, time: float | Tensor, *, interpolate: bool | None = None) -> Tensor:
        interpolate = self.interpolate if interpolate is None else interpolate
        state_cpu, orig_shape, orig_dtype, orig_device = self._prepare_state(state)
        time_cpu = self._prepare_time(time, state_cpu.shape[0])
        idx_state, idx_time = self._index(state_cpu, time_cpu, interpolate)

        flat_idx = _flatten_multi_index_util(idx_state, self._state_strides)
        grads_flat = self._grads.reshape(-1, self.num_times, self.state_dim)
        out = grads_flat[flat_idx, idx_time].reshape(*orig_shape, self.state_dim)
        return out.to(device=orig_device, dtype=orig_dtype)

    # ------------------------------------------------------------------ #
    # Channel extraction helpers
    # ------------------------------------------------------------------ #
    def optimal_uncertainty(self, state: Tensor, time: float | Tensor, *, interpolate: bool | None = None) -> Tensor:
        if self.uncertainty_given:
            raise RuntimeError("Uncertainty channel is GIVEN; optimisation results unavailable.")

        grad = self.gradient(state, time, interpolate=interpolate)
        hook = self.hj_dynamics.optimal_uncertainty_from_grad
        
        # Handle batched time: if time is a tensor with multiple unique values,
        # call hook separately for each unique time to avoid using only time[0]
        if isinstance(time, torch.Tensor) and time.numel() > 1:
            state_flat = state.reshape(-1, state.shape[-1])
            grad_flat = grad.reshape(-1, grad.shape[-1])
            time_flat = time.reshape(-1)
            
            result = torch.zeros_like(state_flat)
            unique_times, inverse_indices = torch.unique(time_flat, return_inverse=True)
            
            for i, t_val in enumerate(unique_times):
                mask = inverse_indices == i
                if mask.any():
                    result[mask] = hook(state_flat[mask], float(t_val.item()), grad_flat[mask])
            
            result = result.reshape_as(state)
        else:
            result = hook(state, float(self._time_scalar(time)), grad)
        
        return torch.as_tensor(result, dtype=state.dtype, device=state.device)

    def optimal_control(self, state: Tensor, time: float | Tensor, *, interpolate: bool | None = None) -> Tensor:
        if self.control_given:
            raise ValueError("Control channel is GIVEN; optimisation results unavailable.")
        hook = self.hj_dynamics.optimal_control_from_grad
        grad = self.gradient(state, time, interpolate=interpolate)
        
        # Handle batched time: if time is a tensor with multiple unique values,
        # call hook separately for each unique time to avoid using only time[0]
        if isinstance(time, torch.Tensor) and time.numel() > 1:
            state_flat = state.reshape(-1, state.shape[-1])
            grad_flat = grad.reshape(-1, grad.shape[-1])
            time_flat = time.reshape(-1)
            
            # Infer control dimension from first call
            first_result = hook(state_flat[:1], float(time_flat[0].item()), grad_flat[:1])
            control_dim = first_result.shape[-1]
            result = torch.zeros((state_flat.shape[0], control_dim), dtype=state.dtype, device=state.device)
            result[:1] = first_result
            
            unique_times, inverse_indices = torch.unique(time_flat, return_inverse=True)
            
            for i in range(1, len(unique_times)):
                t_val = unique_times[i]
                mask = inverse_indices == i
                if mask.any():
                    result[mask] = hook(state_flat[mask], float(t_val.item()), grad_flat[mask])
            
            result = result.reshape(*state.shape[:-1], control_dim)
        else:
            result = hook(state, float(self._time_scalar(time)), grad)
        
        return torch.as_tensor(result, dtype=state.dtype, device=state.device)

    def optimal_disturbance(self, state: Tensor, time: float | Tensor, *, interpolate: bool | None = None) -> Tensor:
        if self.disturbance_given:
            raise ValueError("Disturbance channel is GIVEN; optimisation results unavailable.")
        hook = self.hj_dynamics.optimal_disturbance_from_grad
        grad = self.gradient(state, time, interpolate=interpolate)
        
        # Handle batched time: if time is a tensor with multiple unique values,
        # call hook separately for each unique time to avoid using only time[0]
        if isinstance(time, torch.Tensor) and time.numel() > 1:
            state_flat = state.reshape(-1, state.shape[-1])
            grad_flat = grad.reshape(-1, grad.shape[-1])
            time_flat = time.reshape(-1)
            
            # Infer disturbance dimension from first call
            first_result = hook(state_flat[:1], float(time_flat[0].item()), grad_flat[:1])
            dist_dim = first_result.shape[-1]
            result = torch.zeros((state_flat.shape[0], dist_dim), dtype=state.dtype, device=state.device)
            result[:1] = first_result
            
            unique_times, inverse_indices = torch.unique(time_flat, return_inverse=True)
            
            for i in range(1, len(unique_times)):
                t_val = unique_times[i]
                mask = inverse_indices == i
                if mask.any():
                    result[mask] = hook(state_flat[mask], float(t_val.item()), grad_flat[mask])
            
            result = result.reshape(*state.shape[:-1], dist_dim)
        else:
            result = hook(state, float(self._time_scalar(time)), grad)
        
        return torch.as_tensor(result, dtype=state.dtype, device=state.device)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _prepare_state(self, state: Tensor):
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        orig_device = state.device
        orig_dtype = state.dtype
        state_cpu = state.detach().cpu().reshape(-1, self.state_dim).to(torch.float32)
        orig_shape = list(state.shape[:-1])
        return state_cpu, orig_shape, orig_dtype, orig_device

    def _prepare_time(self, time: float | Tensor, batch: int) -> Tensor:
        if isinstance(time, torch.Tensor):
            flat = time.detach().cpu().reshape(-1).to(torch.float32)
            if flat.numel() == 1:
                return flat.new_full((batch,), float(flat.item()))
            return flat
        return torch.full((batch,), float(time), dtype=torch.float32)

    def _index(self, state_cpu: Tensor, time_cpu: Tensor, interpolate: bool):
        if interpolate:
            idx_state = _nearest_state_indices_util(self._axes, state_cpu).to(torch.int64)
            idx_time = _nearest_time_index_util(self._times, time_cpu).to(torch.int64)
        else:
            idx_state = _exact_state_indices_util(self._axes, state_cpu).to(torch.int64)
            idx_time = _exact_axis_indices_util(self._times, time_cpu).to(torch.int64)
        return idx_state, idx_time

    def _time_scalar(self, time: float | Tensor) -> float:
        if isinstance(time, torch.Tensor):
            return float(time.reshape(-1)[0].detach().cpu().item())
        return float(time)

    def _resolve_system(self) -> System | None:
        if self._system_cache is not None:
            return self._system_cache

        system = getattr(self.hj_dynamics, "system_instance", None)
        if isinstance(system, System):
            self._system_cache = system
            return system

        system_attr = getattr(self.hj_dynamics, "system", None)
        if isinstance(system_attr, System):
            self._system_cache = system_attr
            return system_attr
        if isinstance(system_attr, type) and issubclass(system_attr, System):
            self._system_cache = system_attr()
            return self._system_cache

        system_name = self.metadata.get("system")
        if system_name:
            cls = get_system_class(system_name)
            if cls is not None:
                self._system_cache = cls()
                return self._system_cache
        return None
