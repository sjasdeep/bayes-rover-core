"""
GridSet - Query-only grid-based input set wrapper

This class represents precomputed input sets (box or hull) over a state-time grid.
The class provides fast queries with nearest-neighbor grid lookup (0th-order interpolation).

Constructor contract:
- set_type: 'box' | 'hull'
- state_grid_points: list[Tensor] of length state_dim, each 1D monotonically increasing
- time_grid_points: 1D Tensor or None (time-invariant)
- For box: box_lower, box_upper tensors shaped [n1,...,nk, nt, input_dim]
- For hull: hull_vertices: list of Tensors per grid cell/time, each [n_vertices, input_dim]
- interpolate: if False, off-grid queries raise errors

Query methods (nearest-neighbor):
- as_box(state[, time]) -> (lower, upper)
- argmax_support(direction, state[, time])
"""

from typing import List, Optional, Tuple, Literal

import torch

from ...core.sets import Set
from ...utils.grids import (
    compute_strides as _compute_strides_util,
    nearest_state_indices as _nearest_state_indices_util,
    nearest_time_index as _nearest_time_index_util,
)

__all__ = ["GridSet"]


class GridSet(Set):
    def __init__(
        self,
        *,
        set_type: Literal['box', 'hull'],
        state_grid_points: List[torch.Tensor],
        time_grid_points: Optional[torch.Tensor] = None,
        box_lower: Optional[torch.Tensor] = None,
        box_upper: Optional[torch.Tensor] = None,
        hull_vertices: Optional[List[torch.Tensor]] = None,
        # Optional pre-padded hull data (preferred for performance/loading)
        hull_vertices_padded: Optional[torch.Tensor] = None,
        hull_vertices_mask: Optional[torch.Tensor] = None,
        hull_state_idx_padded: Optional[torch.Tensor] = None,
        # Provenance for box sets: per-corner flat state indices (int tensor)
        box_state_est_corner_idx: Optional[torch.Tensor] = None,
        interpolate: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.set_type = set_type
        self._state_grid_points = state_grid_points
        self._time_grid_points = time_grid_points
        self.interpolate = bool(interpolate)
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Data payloads
        self._box_lower = box_lower
        self._box_upper = box_upper
        self._hull_vertices = hull_vertices
        # If padded hull tensors were provided, keep them directly
        self._hull_padded = hull_vertices_padded
        self._hull_mask = hull_vertices_mask
        # Per-vertex provenance for hull sets: flat state indices padded [n_cells, Vmax]
        self._hull_state_idx_padded = hull_state_idx_padded
        self._box_state_est_corner_idx = box_state_est_corner_idx  # [n1..nk, nt, 2^U]

        # Derived
        self._system = None

        # Basic validation
        if self.set_type not in ('box', 'hull'):
            raise ValueError(f"Unsupported set_type: {self.set_type}")
        if not isinstance(self._state_grid_points, list) or len(self._state_grid_points) == 0:
            raise ValueError("state_grid_points must be a non-empty list of 1D tensors")
        if self.set_type == 'box':
            if self._box_lower is None or self._box_upper is None:
                raise ValueError("Box set requires box_lower and box_upper tensors")
        else:  # hull
            # Accept either explicit per-cell lists or pre-padded hull tensors
            if self._hull_vertices is None and (self._hull_padded is None or self._hull_mask is None):
                raise ValueError("Hull set requires hull_vertices list or padded hull tensors (hull_vertices_padded + hull_vertices_mask)")

        # Move data to device
        self.to(self.device)

        # Lazy hull padding placeholders if not provided
        if self._hull_padded is None:
            self._hull_padded = None  # [n_cells, Vmax, input_dim]
        if self._hull_mask is None:
            self._hull_mask = None    # [n_cells, Vmax] bool

    # --- Internal utilities -------------------------------------------------

    # Wiring helpers
    def bind(self, system) -> None:
        self._system = system

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        for i in range(len(self._state_grid_points)):
            self._state_grid_points[i] = self._state_grid_points[i].to(self.device)
        if self._time_grid_points is not None:
            self._time_grid_points = self._time_grid_points.to(self.device)
        if self._box_lower is not None:
            self._box_lower = self._box_lower.to(self.device)
        if self._box_upper is not None:
            self._box_upper = self._box_upper.to(self.device)
        if self._hull_vertices is not None:
            self._hull_vertices = [v.to(self.device) for v in self._hull_vertices]
        if self._hull_padded is not None:
            self._hull_padded = self._hull_padded.to(self.device)
        if self._hull_mask is not None:
            self._hull_mask = self._hull_mask.to(self.device)
        if getattr(self, "_hull_state_idx_padded", None) is not None and self._hull_state_idx_padded is not None:
            self._hull_state_idx_padded = self._hull_state_idx_padded.to(self.device)
        if getattr(self, "_box_state_est_corner_idx", None) is not None and self._box_state_est_corner_idx is not None:
            self._box_state_est_corner_idx = self._box_state_est_corner_idx.to(self.device)
        return self

    # Shapes/utilities
    @property
    def state_dim(self) -> int:
        return len(self._state_grid_points)

    @property
    def input_dim(self) -> int:
        if self.set_type == 'box' and self._box_lower is not None:
            return int(self._box_lower.shape[-1])
        if self._hull_vertices and len(self._hull_vertices) > 0:
            return int(self._hull_vertices[0].shape[-1])
        if self._hull_padded is not None and self._hull_padded.numel() > 0:
            return int(self._hull_padded.shape[-1])
        return 0

    @property
    def time_dim(self) -> int:
        return 0 if self._time_grid_points is None else int(self._time_grid_points.numel())

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        base = tuple(len(ax) for ax in self._state_grid_points)
        return base + ((self.time_dim,) if self.time_dim > 0 else ())

    # Index helpers (snap/interpolate)

    def _compute_strides(self) -> Tuple[torch.Tensor, int]:
        """Compute state-axis strides (last axis fastest) and n_states.

        Returns:
            (strides_tensor [state_dim], n_states int)
        """
        axis_lengths = [len(ax) for ax in self._state_grid_points]
        strides = _compute_strides_util(axis_lengths)
        n_states = int(torch.tensor(axis_lengths, dtype=torch.long).prod().item()) if axis_lengths else 0
        return torch.tensor(strides, device=self.device, dtype=torch.long), n_states

    # Removed trivial wrapper around shared nearest-axis utility to reduce bloat

    # Removed trivial wrappers around shared utilities (nearest state/time index)

    def _ensure_hull_padded(self) -> None:
        if self._hull_padded is not None:
            return
        if self._hull_vertices is None or len(self._hull_vertices) == 0:
            # Create trivial empty padding
            n_cells = 0
            if self._time_grid_points is None and self._state_grid_points:
                n_cells = 1
            if self._hull_vertices is not None:
                n_cells = len(self._hull_vertices)
            self._hull_padded = torch.zeros(int(n_cells), 1, self.input_dim, device=self.device)
            self._hull_mask = torch.zeros(int(n_cells), 1, dtype=torch.bool, device=self.device)
            return
        Vmax = max(int(v.shape[0]) for v in self._hull_vertices)
        if Vmax <= 0:
            n_cells = len(self._hull_vertices)
            self._hull_padded = torch.zeros(int(n_cells), 1, self.input_dim, device=self.device)
            self._hull_mask = torch.zeros(int(n_cells), 1, dtype=torch.bool, device=self.device)
            return
        n_cells = len(self._hull_vertices)
        pad = torch.zeros(n_cells, Vmax, self.input_dim, device=self.device)
        mask = torch.zeros(n_cells, Vmax, dtype=torch.bool, device=self.device)
        for i, verts in enumerate(self._hull_vertices):
            nv = int(verts.shape[0])
            if nv > 0:
                pad[i, :nv, :] = verts.to(self.device)
                mask[i, :nv] = True
        self._hull_padded = pad
        self._hull_mask = mask

    def _ensure_hull_box_from_vertices(self) -> None:
        """For hull sets, lazily derive per-cell axis-aligned box tensors from vertices.

        Produces self._box_lower/_box_upper shaped [n1,...,nk, nt, input_dim].
        """
        if self.set_type != 'hull':
            return
        if self._box_lower is not None and self._box_upper is not None:
            return
        if self._hull_vertices is None and self._hull_padded is None:
            raise ValueError("Hull set has no vertices to derive box bounds from")

        # Determine sizes
        state_sizes = [len(ax) for ax in self._state_grid_points]
        t_size = (self._time_grid_points.numel() if self._time_grid_points is not None else 1)
        input_dim = self.input_dim

        # If we have padded hull tensors, compute per-cell min/max using mask
        if self._hull_padded is not None and self._hull_mask is not None:
            V = self._hull_padded.shape[1]
            dtype = self._hull_padded.dtype
            # Replace invalid vertices with +inf/-inf so min/max ignore them
            finfo = torch.finfo(dtype)
            masked_min = torch.where(self._hull_mask.unsqueeze(-1), self._hull_padded, torch.full_like(self._hull_padded, finfo.max))
            masked_max = torch.where(self._hull_mask.unsqueeze(-1), self._hull_padded, torch.full_like(self._hull_padded, finfo.min))
            per_cell_min = masked_min.amin(dim=1)  # [n_cells, input_dim]
            per_cell_max = masked_max.amax(dim=1)  # [n_cells, input_dim]
            shaped = (*state_sizes, int(t_size), int(input_dim))
            self._box_lower = per_cell_min.reshape(shaped)
            self._box_upper = per_cell_max.reshape(shaped)
            return

        # Otherwise derive from explicit vertex lists
        # Pick dtype from first non-empty vertex set
        dtype = None
        for v in self._hull_vertices:
            if v is not None and v.numel() > 0:
                dtype = v.dtype
                break
        if dtype is None:
            dtype = torch.float32

        lowers: List[torch.Tensor] = []
        uppers: List[torch.Tensor] = []
        for verts in self._hull_vertices:
            if verts is None or verts.numel() == 0:
                lowers.append(torch.zeros(input_dim, device=self.device, dtype=dtype))
                uppers.append(torch.zeros(input_dim, device=self.device, dtype=dtype))
            else:
                v = verts.to(self.device)
                lowers.append(torch.min(v, dim=0)[0])
                uppers.append(torch.max(v, dim=0)[0])

        stacked_l = torch.stack(lowers, dim=0)
        stacked_u = torch.stack(uppers, dim=0)
        shaped = (*state_sizes, int(t_size), int(input_dim))
        self._box_lower = stacked_l.reshape(shaped)
        self._box_upper = stacked_u.reshape(shaped)

    

    # Public API
    def as_box(
        self,
        state: torch.Tensor,
        time: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.set_type == 'hull':
            # Lazily derive box tensors from hull vertices, then reuse vectorized gather
            self._ensure_hull_box_from_vertices()
        # Move to device and handle batch
        orig_device = state.device
        state_b = state.to(self.device)
        single = False
        if state_b.ndim == 1:
            state_b = state_b.unsqueeze(0)
            single = True

        # Compute strides
        strides_tensor, n_states = self._compute_strides()

        # Vectorized nearest indices per dim and time
        idx_state = _nearest_state_indices_util(self._state_grid_points, state_b).to(torch.long)  # [B, state_dim]
        t_idx = int(_nearest_time_index_util(self._time_grid_points, float(time))[0].item()) if self._time_grid_points is not None else 0

        # Off-grid enforcement when interpolate=False
        if not self.interpolate:
            tol = 1e-6
            off_any = torch.zeros(idx_state.shape[0], dtype=torch.bool, device=self.device)
            bad_dim = None
            for d, axis in enumerate(self._state_grid_points):
                nearest_vals = axis[idx_state[:, d]]
                diffs = torch.abs(state_b[:, d].to(axis.dtype) - nearest_vals)
                off = diffs > tol
                if off.any() and bad_dim is None:
                    bad_dim = d
                off_any |= off
            if self._time_grid_points is not None:
                t_tensor = torch.tensor(time, device=self.device, dtype=self._time_grid_points.dtype)
                t_nearest = self._time_grid_points[t_idx]
                off_t = torch.abs(t_tensor - t_nearest) > tol
                if off_t:
                    off_any |= torch.ones_like(off_any, dtype=torch.bool, device=self.device)
            if off_any.any():
                if bad_dim is None and self._time_grid_points is not None:
                    raise ValueError(
                        f"GridSet.as_box: query time off-grid (t={float(time):.6f}, nearest={float(self._time_grid_points[t_idx].item()):.6f}). "
                        f"Set interpolate=True to allow nearest-neighbor queries."
                    )
                # Report first offending example for clarity
                first = int(torch.nonzero(off_any, as_tuple=False)[0].item())
                d = bad_dim if bad_dim is not None else 0
                axis = self._state_grid_points[d]
                nearest_val = axis[idx_state[first, d]]
                val = state_b[first, d]
                raise ValueError(
                    f"GridSet.as_box: query off-grid for dim {d} (value={float(val.item()):.6f}, "
                    f"nearest={float(nearest_val.item()):.6f}). Set interpolate=True to allow nearest-neighbor."
                )

        # Compute flat indices for gather using memory-contiguous order.
        # When viewing as _box_lower.view(-1, input_dim), the time axis is interleaved
        # within the linearized state index. Therefore, multiply state strides by T.
        B = idx_state.shape[0]
        T = (self.time_dim if self._time_grid_points is not None else 1)
        state_strides_for_view = strides_tensor.view(1, -1) * int(T)
        flat = (idx_state.to(torch.long) * state_strides_for_view).sum(dim=1) + int(t_idx)

        # Gather from flattened box tensors
        input_dim = self.input_dim
        lower_flat = self._box_lower.view(-1, input_dim)
        upper_flat = self._box_upper.view(-1, input_dim)
        lower = lower_flat[flat].to(orig_device)
        upper = upper_flat[flat].to(orig_device)
        if single:
            lower = lower.squeeze(0)
            upper = upper.squeeze(0)
        return lower, upper

    def argmax_support(
        self,
        direction: torch.Tensor,
        state: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        # Hull sets: compute support against true hull vertices (vectorized using padded hulls)
        if self.set_type == 'hull' and (self._hull_vertices is not None or self._hull_padded is not None):
            device = self.device
            # Prepare batch shapes
            state_b = state.to(device)
            dir_b = direction.to(device)
            if state_b.ndim == 1:
                state_b = state_b.unsqueeze(0)
            if dir_b.ndim == 1:
                dir_b = dir_b.unsqueeze(0)
            if state_b.shape[0] != dir_b.shape[0]:
                raise ValueError("argmax_support: batch size mismatch between state and direction")

            # Strides and indices
            idx_state = _nearest_state_indices_util(self._state_grid_points, state_b).to(torch.long)
            strides_tensor, n_states = self._compute_strides()
            t_idx = int(_nearest_time_index_util(self._time_grid_points, float(time))[0].item()) if self._time_grid_points is not None else 0

            # Off-grid enforcement if interpolate=False
            if not self.interpolate:
                tol = 1e-6
                off_any = torch.zeros(idx_state.shape[0], dtype=torch.bool, device=device)
                for d, axis in enumerate(self._state_grid_points):
                    nearest_vals = axis[idx_state[:, d]]
                    diffs = torch.abs(state_b[:, d].to(axis.dtype) - nearest_vals)
                    off_any |= (diffs > tol)
                if self._time_grid_points is not None:
                    t_tensor = torch.tensor(time, device=device, dtype=self._time_grid_points.dtype)
                    t_nearest = self._time_grid_points[t_idx]
                    if torch.abs(t_tensor - t_nearest) > tol:
                        off_any |= torch.ones_like(off_any, dtype=torch.bool, device=device)
                if off_any.any():
                    first = int(torch.nonzero(off_any, as_tuple=False)[0].item())
                    raise ValueError(
                        "GridSet.argmax_support: one or more queries are off-grid; set interpolate=True to allow nearest-neighbor."
                    )

            # Build flat indices [B] consistent with view(-1, input_dim)
            B = idx_state.shape[0]
            T = (self.time_dim if self._time_grid_points is not None else 1)
            state_strides_for_view = strides_tensor.view(1, -1) * int(T)
            flat_idx = (idx_state.to(torch.long) * state_strides_for_view).sum(dim=1) + int(t_idx)

            # Pad hulls lazily and gather per-batch cell hulls
            self._ensure_hull_padded()
            cell_verts = self._hull_padded[flat_idx]  # [B, Vmax, input_dim]
            cell_mask = self._hull_mask[flat_idx]     # [B, Vmax]

            # Compute support: argmax over vertices
            # dots: [B, Vmax]
            dots = (cell_verts * dir_b.unsqueeze(1)).sum(dim=-1)
            # mask invalid vertices to -inf
            finfo_min = torch.finfo(dots.dtype).min
            dots_masked = torch.where(cell_mask, dots, torch.full_like(dots, finfo_min))
            max_idx = torch.argmax(dots_masked, dim=1)  # [B]
            # For rows with no valid vertices, return zeros
            has_any = cell_mask.any(dim=1)
            gathered = cell_verts[torch.arange(B, device=device), max_idx]
            gathered = torch.where(
                has_any.unsqueeze(1),
                gathered,
                torch.zeros(B, self.input_dim, device=device, dtype=state_b.dtype),
            )
            return gathered if direction.ndim > 1 else gathered.squeeze(0)

        # Default (box or hull-without-vertices): use box extremes per sign
        lower, upper = self.as_box(state, time)
        if direction.ndim == 1:
            direction_b = direction.unsqueeze(0)
        else:
            direction_b = direction.reshape(-1, direction.shape[-1])
        choice = torch.where(direction_b >= 0, upper, lower)
        return choice if direction.ndim > 1 else choice.squeeze(0)

    def argmax_support_with_state_est(
        self,
        direction: torch.Tensor,
        state: torch.Tensor,
        time: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Like argmax_support, but also returns the producing estimated state x_hat when available.

    Returns (u_star, xhat_star or None). For hull sets, xhat_star is returned if
    hull_state_estimates were provided. For box sets, xhat_star is returned if
    per-corner provenance (box_state_est_corners) is available.
        """
        if self.set_type == 'hull' and (self._hull_vertices is not None or self._hull_padded is not None):
            device = self.device
            state_b = state.to(device)
            dir_b = direction.to(device)
            single = False
            if state_b.ndim == 1:
                state_b = state_b.unsqueeze(0)
                dir_b = dir_b.unsqueeze(0) if dir_b.ndim == 1 else dir_b
                single = True
            idx_state = _nearest_state_indices_util(self._state_grid_points, state_b).to(torch.long)
            strides_tensor, n_states = self._compute_strides()
            t_idx = int(_nearest_time_index_util(self._time_grid_points, float(time))[0].item()) if self._time_grid_points is not None else 0
            B = idx_state.shape[0]
            T = (self.time_dim if self._time_grid_points is not None else 1)
            state_strides_for_view = strides_tensor.view(1, -1) * int(T)
            flat_idx = (idx_state.to(torch.long) * state_strides_for_view).sum(dim=1) + int(t_idx)
            self._ensure_hull_padded()
            cell_verts = self._hull_padded[flat_idx]
            cell_mask = self._hull_mask[flat_idx]
            dots = (cell_verts * dir_b.unsqueeze(1)).sum(dim=-1)
            finfo_min = torch.finfo(dots.dtype).min
            dots_masked = torch.where(cell_mask, dots, torch.full_like(dots, finfo_min))
            max_idx = torch.argmax(dots_masked, dim=1)
            u_star = cell_verts[torch.arange(B, device=device), max_idx]
            # Reconstruct xhat from per-vertex flat state indices if available
            if getattr(self, "_hull_state_idx_padded", None) is not None and self._hull_state_idx_padded is not None:
                idx_pad = self._hull_state_idx_padded[flat_idx]  # [B, Vmax]
                state_idx_flat = idx_pad[torch.arange(B, device=device), max_idx].to(torch.long)  # [B]
                strides_tensor, n_states = self._compute_strides()
                rem = state_idx_flat.clone()
                idxs_per_dim = []
                for d in range(self.state_dim):
                    stride_d = strides_tensor[d]
                    idxd = rem // stride_d
                    rem = rem % stride_d
                    idxs_per_dim.append(idxd)
                xhat_cols = []
                for d in range(self.state_dim):
                    axis = self._state_grid_points[d]
                    xhat_cols.append(axis[idxs_per_dim[d]])
                xhat_star = torch.stack(xhat_cols, dim=1)
            else:
                xhat_star = None
            if single:
                u_star = u_star.squeeze(0)
                if xhat_star is not None:
                    xhat_star = xhat_star.squeeze(0)
            return u_star, xhat_star

        # Box logic
        lower, upper = self.as_box(state, time)
        if direction.ndim == 1:
            direction_b = direction.unsqueeze(0)
            single = True
        else:
            direction_b = direction.reshape(-1, direction.shape[-1])
            single = False
        choice = torch.where(direction_b >= 0, upper, lower)
        xhat = None
        # Prefer per-corner provenance via flat state indices if available
        if getattr(self, "_box_state_est_corner_idx", None) is not None and self._box_state_est_corner_idx is not None:
            # Compute flat indices to gather the per-corner state flat indices for each batch row
            state_b = state.to(self.device)
            if state_b.ndim == 1:
                state_b = state_b.unsqueeze(0)
            strides_tensor, n_states = self._compute_strides()
            idx_state = _nearest_state_indices_util(self._state_grid_points, state_b).to(torch.long)
            t_idx = int(_nearest_time_index_util(self._time_grid_points, float(time))[0].item()) if self._time_grid_points is not None else 0
            T = (self.time_dim if self._time_grid_points is not None else 1)
            state_strides_for_view = strides_tensor.view(1, -1) * int(T)
            flat = (idx_state.to(torch.long) * state_strides_for_view).sum(dim=1) + int(t_idx)
            # Corner code from direction signs
            signs = (direction_b >= 0).to(torch.long)  # [B, U]
            U = signs.shape[-1]
            weights = (2 ** torch.arange(U, device=self.device, dtype=torch.long))
            corner_code = (signs * weights.view(1, -1)).sum(dim=-1)  # [B]
            # Gather [B, Nc] int indices
            Nc = int(self._box_state_est_corner_idx.shape[-1])
            corner_idx_tensor = self._box_state_est_corner_idx.view(-1, Nc)[flat]  # [B, Nc]
            state_idx_flat = corner_idx_tensor[torch.arange(corner_idx_tensor.shape[0], device=self.device), corner_code].to(torch.long)  # [B]
            # Reconstruct per-dim indices via strides
            Bn = state_idx_flat.shape[0]
            rem = state_idx_flat.clone()
            idxs_per_dim = []
            for d in range(self.state_dim):
                stride_d = strides_tensor[d]
                idxd = rem // stride_d
                rem = rem % stride_d
                idxs_per_dim.append(idxd)
            # Build xhat by gathering axis values
            xhat_cols = []
            for d in range(self.state_dim):
                axis = self._state_grid_points[d]
                xhat_cols.append(axis[idxs_per_dim[d]])
            xhat = torch.stack(xhat_cols, dim=1)  # [B, D]
        if single:
            choice = choice.squeeze(0)
            if xhat is not None:
                xhat = xhat.squeeze(0)
        return choice, xhat
