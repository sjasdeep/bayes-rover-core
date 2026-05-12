"""
JAX-native GridSet wrapper with JIT-friendly queries.

This adapter preloads data from the PyTorch GridSet into JAX arrays and exposes
the Set interface for the solver: as_box(state, time) and argmax_support(direction, state, time).
All computations are implemented with jax.numpy for JIT compatibility.
"""

from typing import Tuple, List

import jax.numpy as jnp
import numpy as np

__all__ = ["JaxGridSet"]


class JaxGridSet:
    """JIT-friendly wrapper on top of a PyTorch GridSet.

    Methods:
      - as_box(state, time) -> (lower, upper)  [JAX arrays]
      - argmax_support(direction, state, time) -> optimal vertex [JAX array]

    Input shapes:
      - state: [D] or [B, D]
      - direction: [U] or [B, U] (U = input_dim)
      Returns match batching of inputs.
    """

    def __init__(self, torch_grid_set) -> None:
        self.set_type = torch_grid_set.set_type

        # Axes and sizes
        self.grid_axes: List[jnp.ndarray] = [self._to_jax(ax) for ax in torch_grid_set._state_grid_points]
        self.state_dim = len(self.grid_axes)
        self.axis_sizes = [int(ax.shape[0]) for ax in self.grid_axes]
        self.n_states = int(np.prod(np.array(self.axis_sizes))) if self.axis_sizes else 1

        # Time axis (may be None => size 1 with dummy value)
        if getattr(torch_grid_set, "_time_grid_points", None) is not None:
            self.grid_times = self._to_jax(torch_grid_set._time_grid_points)
        else:
            self.grid_times = jnp.array([0.0])
        self.nt = int(self.grid_times.shape[0])

        # Precompute state strides for row-major flatten over state dims
        # stride[i] = product(axis_sizes[i+1:])
        strides = []
        for i in range(self.state_dim):
            prod = 1
            for s in range(i + 1, self.state_dim):
                prod *= self.axis_sizes[s]
            strides.append(prod)
        self.state_strides = jnp.array(strides, dtype=jnp.int32) if strides else jnp.array([], dtype=jnp.int32)

        # Load payload
        if self.set_type == 'box':
            lower = torch_grid_set._box_lower  # [n1..nk, nt, U]
            upper = torch_grid_set._box_upper
            if lower is None or upper is None:
                raise ValueError("Box set requires lower/upper tensors")
            self.input_dim = int(lower.shape[-1])
            # Convert to JAX
            self.lower = self._to_jax(lower)  # shape [n1..nk, nt, U]
            self.upper = self._to_jax(upper)
            self.box_state_corner_idx = None
            if getattr(torch_grid_set, "_box_state_est_corner_idx", None) is not None and torch_grid_set._box_state_est_corner_idx is not None:
                self.box_state_corner_idx = self._to_jax(torch_grid_set._box_state_est_corner_idx)
        elif self.set_type == 'hull':
            # Prefer pre-padded tensors if available on the torch GridSet
            if getattr(torch_grid_set, "_hull_padded", None) is not None and torch_grid_set._hull_padded is not None:
                hp = torch_grid_set._hull_padded.detach().cpu().numpy()
                hm = (torch_grid_set._hull_mask.detach().cpu().numpy()
                      if getattr(torch_grid_set, "_hull_mask", None) is not None and torch_grid_set._hull_mask is not None else None)
                hi = (torch_grid_set._hull_state_idx_padded.detach().cpu().numpy()
                      if getattr(torch_grid_set, "_hull_state_idx_padded", None) is not None and torch_grid_set._hull_state_idx_padded is not None else None)
                self.input_dim = int(hp.shape[-1])
                self.hulls = jnp.array(hp)
                self.hulls_mask = jnp.array(hm) if hm is not None else jnp.ones(hp.shape[:2], dtype=bool)
                self.hulls_state_idx = jnp.array(hi) if hi is not None else None
            else:
                verts_list = torch_grid_set._hull_vertices
                if verts_list is None:
                    raise ValueError("Hull set requires vertices list")
                # Convert list to JAX arrays and pad
                np_list = [v.detach().cpu().numpy() for v in verts_list]
                if len(np_list) == 0:
                    raise ValueError("Hull vertices list is empty")
                self.input_dim = int(np_list[0].shape[1]) if np_list[0].ndim == 2 else 1
                vcounts = [arr.shape[0] for arr in np_list]
                vmax = max(vcounts)
                padded = []
                masks = []
                for arr, n in zip(np_list, vcounts):
                    pad = np.zeros((vmax - n, self.input_dim), dtype=arr.dtype)
                    padded.append(np.vstack([arr, pad]))
                    mask = np.concatenate([np.ones((n,), dtype=bool), np.zeros((vmax - n,), dtype=bool)])
                    masks.append(mask)
                self.hulls = jnp.stack([jnp.array(x) for x in padded])              # [Ncells, Vmax, U]
                self.hulls_mask = jnp.stack([jnp.array(m) for m in masks])          # [Ncells, Vmax]
                # No per-vertex indices available in this path
                self.hulls_state_idx = None
        else:
            raise ValueError(f"Unknown set_type: {self.set_type}")

    def _to_jax(self, t) -> jnp.ndarray:
        return jnp.array(np.asarray(t.detach().cpu().numpy()))

    def _nearest_state_indices(self, state_b: jnp.ndarray) -> jnp.ndarray:
        # state_b: [B, D]
        B = state_b.shape[0]
        idxs = []
        for d in range(self.state_dim):
            axis = self.grid_axes[d]  # [Nd]
            # distances: [B, Nd]
            dists = jnp.abs(state_b[:, d:d+1] - axis[None, :])
            idxs.append(jnp.argmin(dists, axis=1))  # [B]
        return jnp.stack(idxs, axis=1)  # [B, D]

    def _nearest_time_index(self, time: float) -> jnp.ndarray:
        # Return scalar index
        dists = jnp.abs(self.grid_times - time)
        return jnp.argmin(dists)

    def _flatten_state_indices(self, idx_state: jnp.ndarray) -> jnp.ndarray:
        # idx_state: [B, D]
        if self.state_dim == 0:
            return jnp.zeros((idx_state.shape[0],), dtype=jnp.int32)
        return jnp.sum(idx_state * self.state_strides[None, :], axis=1).astype(jnp.int32)

    def as_box(self, state: jnp.ndarray, time: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Normalize batch
        state_b = state[None, :] if state.ndim == 1 else state
        # Indices
        idx_state = self._nearest_state_indices(state_b)             # [B, D]
        t_idx = self._nearest_time_index(time).astype(jnp.int32)     # scalar

        if self.set_type == 'box':
            # Direct gather using row-major flatten over [n1..nk, nt]
            B = idx_state.shape[0]
            state_flat = self._flatten_state_indices(idx_state)      # [B]
            # row-major flatten index: state_idx * nt + t_idx
            flat = state_flat * self.nt + t_idx                      # [B]
            lo_flat = self.lower.reshape((-1, self.input_dim))
            hi_flat = self.upper.reshape((-1, self.input_dim))
            lo = lo_flat[flat]
            hi = hi_flat[flat]
            return (lo[0] if state.ndim == 1 else lo, 
                    hi[0] if state.ndim == 1 else hi)
        else:  # hull -> return AABB via per-row min/max
            B = idx_state.shape[0]
            state_flat = self._flatten_state_indices(idx_state)
            flat = state_flat * self.nt + t_idx
            verts = self.hulls[flat]        # [B, Vmax, U] or [Vmax, U] if B==1
            mask = self.hulls_mask[flat]    # [B, Vmax] or [Vmax]
            if state.ndim == 1:
                # Single
                valid = mask
                # Replace invalids with +inf/-inf to compute min/max over valid only
                big = 1e30
                verts_min = jnp.where(valid[:, None], verts, big)
                verts_max = jnp.where(valid[:, None], verts, -big)
                lo = jnp.min(verts_min, axis=0)
                hi = jnp.max(verts_max, axis=0)
                return lo, hi
            else:
                # Batched
                big = 1e30
                verts_min = jnp.where(mask[:, :, None], verts, big)
                verts_max = jnp.where(mask[:, :, None], verts, -big)
                lo = jnp.min(verts_min, axis=1)
                hi = jnp.max(verts_max, axis=1)
                return lo, hi

    def argmax_support(self, direction: jnp.ndarray, state: jnp.ndarray, time: float) -> jnp.ndarray:
        # Normalize batch
        state_b = state[None, :] if state.ndim == 1 else state
        dir_b = direction[None, :] if direction.ndim == 1 else direction

        if self.set_type == 'hull':
            idx_state = self._nearest_state_indices(state_b)
            t_idx = self._nearest_time_index(time).astype(jnp.int32)
            state_flat = self._flatten_state_indices(idx_state)
            flat = state_flat * self.nt + t_idx
            verts = self.hulls[flat]      # [B, Vmax, U]
            mask = self.hulls_mask[flat]  # [B, Vmax]
            # Compute masked dot products
            dots = jnp.einsum('bvu,bu->bv', verts, dir_b)
            neg_inf = -jnp.inf
            masked = jnp.where(mask, dots, neg_inf)
            max_idx = jnp.argmax(masked, axis=1)
            best = verts[jnp.arange(verts.shape[0]), max_idx]
            return best[0] if direction.ndim == 1 else best
        else:
            # Box: choose extreme by sign of direction
            lo, hi = self.as_box(state, time)
            if direction.ndim == 1:
                return jnp.where(direction >= 0, hi, lo)
            else:
                return jnp.where(dir_b >= 0, hi, lo)

    def argmax_support_with_state_est(
        self, direction: jnp.ndarray, state: jnp.ndarray, time: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (u*, xhat*, has_state) with same batching as inputs.

        has_state is a boolean mask indicating whether provenance was available.
        When false, xhat* is a zero array.
        """
        # Normalize batch
        state_b = state[None, :] if state.ndim == 1 else state
        dir_b = direction[None, :] if direction.ndim == 1 else direction
        B = state_b.shape[0]
        if self.set_type == 'hull':
            idx_state = self._nearest_state_indices(state_b)
            t_idx = self._nearest_time_index(time).astype(jnp.int32)
            state_flat = self._flatten_state_indices(idx_state)
            flat = state_flat * self.nt + t_idx
            verts = self.hulls[flat]      # [B, Vmax, U]
            mask = self.hulls_mask[flat]  # [B, Vmax]
            dots = jnp.einsum('bvu,bu->bv', verts, dir_b)
            neg_inf = -jnp.inf
            masked = jnp.where(mask, dots, neg_inf)
            max_idx = jnp.argmax(masked, axis=1)
            best_u = verts[jnp.arange(B), max_idx]
            if getattr(self, 'hulls_state_idx', None) is not None and self.hulls_state_idx is not None:
                idx_pad = self.hulls_state_idx[flat]  # [B, Vmax]
                state_idx_flat = idx_pad[jnp.arange(B), max_idx].astype(jnp.int32)
                # Reconstruct per-dim indices and then axis values
                rem = state_idx_flat
                cols = []
                for d in range(self.state_dim):
                    stride_d = self.state_strides[d] if self.state_dim > 0 else jnp.array(1, dtype=jnp.int32)
                    idxd = rem // stride_d
                    rem = rem % stride_d
                    cols.append(self.grid_axes[d][idxd])
                best_x = jnp.stack(cols, axis=1) if self.state_dim > 0 else jnp.zeros((B, 0))
                has = jnp.ones((B,), dtype=bool)
            else:
                best_x = jnp.zeros((B, self.state_dim))
                has = jnp.zeros((B,), dtype=bool)
            if direction.ndim == 1:
                return best_u[0], best_x[0], has[0]
            return best_u, best_x, has
        else:
            lo, hi = self.as_box(state, time)
            # Compute best_u by sign rule
            best_u = jnp.where((direction if direction.ndim == 1 else dir_b) >= 0, hi, lo)
            # Provenance selection
            if getattr(self, 'box_state_corner_idx', None) is not None and self.box_state_corner_idx is not None:
                idx_state = self._nearest_state_indices(state_b)
                t_idx = self._nearest_time_index(time).astype(jnp.int32)
                state_flat = self._flatten_state_indices(idx_state)
                flat = state_flat * self.nt + t_idx
                corners_idx = self.box_state_corner_idx.reshape((-1, self.box_state_corner_idx.shape[-1]))[flat]  # [B, Nc]
                # Corner code from sign bits: lower=0, upper=1 per dim; LSB corresponds to last dim
                bits = (direction if direction.ndim == 1 else dir_b) >= 0  # [U] or [B, U]
                if direction.ndim == 1:
                    weights = (2 ** jnp.arange(self.input_dim, dtype=jnp.int32))
                    code = jnp.sum(weights * bits.astype(jnp.int32))
                    state_idx_flat = corners_idx[0, code]
                    # Reconstruct xhat from flat state index via strides
                    rem = state_idx_flat
                    cols = []
                    for d in range(self.state_dim):
                        stride_d = self.state_strides[d] if self.state_dim > 0 else jnp.array(1, dtype=jnp.int32)
                        idxd = rem // stride_d
                        rem = rem % stride_d
                        cols.append(self.grid_axes[d][idxd])
                    best_x = jnp.stack(cols) if self.state_dim > 0 else jnp.array([])
                    return best_u, best_x, jnp.array(True)
                else:
                    weights = (2 ** jnp.arange(self.input_dim, dtype=jnp.int32))[None, :]
                    code = jnp.sum(weights * bits.astype(jnp.int32), axis=1)  # [B]
                    state_idx_flat = corners_idx[jnp.arange(corners_idx.shape[0]), code]
                    # Reconstruct per-dim indices
                    rem = state_idx_flat
                    cols = []
                    for d in range(self.state_dim):
                        stride_d = self.state_strides[d] if self.state_dim > 0 else jnp.array(1, dtype=jnp.int32)
                        idxd = rem // stride_d
                        rem = rem % stride_d
                        cols.append(self.grid_axes[d][idxd])
                    best_x = jnp.stack(cols, axis=1) if self.state_dim > 0 else jnp.zeros((B, 0))
                    return best_u, best_x, jnp.ones((B,), dtype=bool)
            # No provenance available
            if direction.ndim == 1:
                return best_u, jnp.zeros((self.state_dim,)), jnp.array(False)
            else:
                return best_u, jnp.zeros((B, self.state_dim)), jnp.zeros((B,), dtype=bool)