#!/usr/bin/env python3
"""
Compare value-based BRTs vs baselines and Monte Carlo under-approximations.

This script overlays:
- Primary GridValue zero-level set (solid black)
- Baseline GridValue zero-level set (dashed black, optional)
- Monte Carlo value snapshots zero contours (rainbow colors, optional)
- Simulation trajectories clipped at first entry into V<=0 w.r.t primary GridValue,
  marking the collision with an 'x' (optional)

Usage examples:
  python scripts/value_evaluation/compare_values.py \
      --value-tag RoverDark_WorstCase \
      --mc-tag my_mc \
      --baseline-tag RoverDark_Nominal \
      --sim-tag RoverDark_Simulation \
      --time 0.0

Notes:
- For 3D+ state spaces, specify which dimension to slice with --slice-dim and optionally --slice-value.
- Obstacles are drawn when plotting the (x,y) plane (dims {0,1}).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Color constants
COLLISION_BLUE = '#0d47a1'  # darker blue for collision trajectories and markers

from src.impl.values.grid_value import GridValue
from src.utils.cache_loaders import (
    load_grid_value_by_tag,
    get_grid_value_metadata,
    instantiate_system_by_name,
)
from src.utils.grids import nearest_axis_indices, nearest_time_index


def _load_mc_cache(tag: str):
    path = Path('.cache') / 'monte_carlo_values' / f'{tag}.pkl'
    if not path.exists():
        raise FileNotFoundError(f"Monte Carlo cache not found: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'tag' not in data:
        data['tag'] = tag
    return data


def _extract_slice_2d(vf: GridValue, time_val: float, dims: Tuple[int, int], slice_dim: Optional[int]=None, slice_value: Optional[float]=None):
    """Return (X, Y, V2d_np, xlabel, ylabel, time_idx, slice_info) for plotting 2D contour of V=0.
    If vf.state_dim==2, dims should be (0,1) and slice_dim is ignored.
    For 3D+, dims are the two plotted dims; slice_dim is the remaining dim to fix.
    """
    # Time index by nearest
    tdiff = torch.abs(vf._times - float(time_val))
    time_idx = int(torch.argmin(tdiff).item())

    # Determine axes
    all_dims = list(range(vf.state_dim))
    i, j = int(dims[0]), int(dims[1])

    if vf.state_dim == 2:
        value_2d = vf._values[..., time_idx]
        X, Y = np.meshgrid(
            vf._axes[i].detach().cpu().numpy(),
            vf._axes[j].detach().cpu().numpy(),
            indexing='ij'
        )
        xlabel, ylabel = 'x', 'y'
    else:
        if slice_dim is None:
            # pick the smallest index not in dims
            rest = [d for d in all_dims if d not in (i, j)]
            if not rest:
                raise ValueError("slice_dim required for state_dim>2 when dims specify only two")
            slice_dim = int(rest[0])
        # find slice index
        axis_t = vf._axes[slice_dim]
        if slice_value is None:
            slice_idx = axis_t.numel() // 2
        else:
            slice_idx = int(nearest_axis_indices(axis_t, torch.tensor([float(slice_value)], dtype=axis_t.dtype, device=axis_t.device))[0].item())
        # build value_2d by moving dims accordingly
        value_nd = vf._values[..., time_idx]
        # order dims to index: we need [:,:,slice_idx] on slice_dim
        indexer = [slice(None)] * vf.state_dim
        indexer[slice_dim] = slice_idx
        value_sliced = value_nd[tuple(indexer)]  # now 2D over remaining dims in original order
        # Map to (i, j) in case their order isn't ascending
        # If slice_dim is before one of (i,j), after indexing it collapses; we need axes X,Y matching i,j
        axes_i = vf._axes[i].detach().cpu().numpy()
        axes_j = vf._axes[j].detach().cpu().numpy()
        X, Y = np.meshgrid(axes_i, axes_j, indexing='ij')
        # Arrange value_sliced to match (i,j) order
        # Determine the positions of i and j in the remaining dims order
        rem = [d for d in all_dims if d != slice_dim]
        if rem != [i, j]:
            # value_sliced currently has dims in order 'rem'
            # We need to transpose to (i,j)
            perm = [rem.index(i), rem.index(j)]
            value_sliced = value_sliced.permute(perm)
        xlabel, ylabel = 'x', 'y' if j == 1 else (f'x{j}')
        # Keep labels generic; we'll set proper labels later via system.state_labels
        value_2d = value_sliced

    V2d_np = value_2d.detach().cpu().numpy() if isinstance(value_2d, torch.Tensor) else np.asarray(value_2d)
    return X, Y, V2d_np, time_idx


def _axis_labels(system, dims: Tuple[int,int]) -> Tuple[str, str]:
    labels = getattr(system, 'state_labels', tuple([f'x{i}' for i in range(len(dims))]))
    i, j = dims
    xi = labels[i] if i < len(labels) else f'x{i}'
    yj = labels[j] if j < len(labels) else f'x{j}'
    return xi, yj


def _plot_obstacles_if_xy(ax, system):
    from src.utils.obstacles import draw_obstacles_2d
    try:
        draw_obstacles_2d(ax, system)
    except Exception:
        pass


def _find_obstacle_entry_indices(system, states: torch.Tensor) -> torch.Tensor:
    """Return first index k (per trajectory) where failure_function(states[:,k,:]) <= 0; -1 if never.
    Uses the system's failure_function (obstacles) rather than GridValue.
    states: [N, T+1, D]
    """
    N, T1, D = states.shape
    hit = torch.full((N,), -1, dtype=torch.int64)
    active = torch.ones((N,), dtype=torch.bool)
    with torch.no_grad():
        for k in range(T1):
            if not active.any():
                break
            xk = states[:, k, :]
            gk = system.failure_function(xk, None)
            if isinstance(gk, torch.Tensor) and gk.ndim > 1:
                gk = gk.squeeze(-1)
            enter = (gk <= 0.0)
            newly = active & enter
            idxs = torch.nonzero(newly, as_tuple=False).reshape(-1)
            if idxs.numel() > 0:
                hit[idxs] = k
                active[idxs] = False
    return hit


def main():
    ap = argparse.ArgumentParser(description='Compare robust GridValue vs baselines, MC snapshots, and trajectories')
    ap.add_argument('--value-tag', type=str, required=True, help='Primary GridValue tag')
    ap.add_argument('--baseline-tag', type=str, default=None, help='Baseline GridValue tag (optional)')
    ap.add_argument('--mc-tag', type=str, default=None, help='Monte Carlo cache tag (optional)')
    ap.add_argument('--sim-tag', type=str, default=None, help='Simulation results tag (optional)')
    ap.add_argument('--sim-tag2', type=str, default=None, help='Secondary simulation tag to overlay trajectories from (optional; non-grid recommended)')
    ap.add_argument('--time', type=float, default=0.0, help='Time value for GridValue contours (default: 0.0)')
    ap.add_argument('--slice-dim', type=int, default=2, help='Slice dimension for 3D+ state spaces (default: 2)')
    ap.add_argument('--slice-value', type=float, default=None, help='Value at which to slice the slice-dim (default: middle)')
    ap.add_argument('--save-dir', type=str, default=None, help='Output directory (default: outputs/visualizations/value_evaluation/{combo})')
    ap.add_argument('--xlim', type=float, nargs=2, default=None, help='x-axis limits [xmin xmax]')
    ap.add_argument('--ylim', type=float, nargs=2, default=None, help='y-axis limits [ymin ymax]')
    ap.add_argument('--interactive', action='store_true', help='Open interactive window instead of saving (if supported)')
    ap.add_argument('--dpi', type=int, default=150, help='Output DPI for saved figure (and interactive window if applicable)')
    ap.add_argument('--hide-est', action='store_true', help='Hide estimated trajectories when overlaying simulations')
    ap.add_argument('--hide-safe', action='store_true', help='Hide safe trajectories (only plot those that collide)')
    args = ap.parse_args()

    if args.interactive:
        try:
            matplotlib.use('TkAgg', force=True)
        except Exception:
            pass
        # Apply requested DPI to interactive windows as well
        try:
            import matplotlib as mpl
            mpl.rcParams['figure.dpi'] = int(args.dpi)
        except Exception:
            pass

    # Load primary GridValue and system
    vf = load_grid_value_by_tag(args.value_tag, interpolate=True)
    meta = get_grid_value_metadata(args.value_tag)
    sys_name = meta.get('system_name', meta.get('system', ''))
    system = instantiate_system_by_name(sys_name) if sys_name else None

    # Determine dims and slice
    if vf.state_dim == 2:
        dims = (0, 1)
        slice_dim = None
    else:
        # choose dims as the first two dims not equal to slice_dim
        sd = int(args.slice_dim)
        dims_all = [d for d in range(vf.state_dim) if d != sd]
        if len(dims_all) < 2:
            raise SystemExit("GridValue has insufficient dims to create a 2D slice; adjust --slice-dim")
        dims = (dims_all[0], dims_all[1])
        slice_dim = sd

    X, Y, V2d, t_idx = _extract_slice_2d(vf, args.time, dims, slice_dim, args.slice_value)

    # Prepare plot
    fig, ax = plt.subplots(figsize=(8, 7))
    # Reserve generous space on the right for an external legend so it isn't clipped
    try:
        fig.subplots_adjust(right=0.68)
    except Exception:
        pass

    # Background: plot primary GridValue actual values as filled contours (symmetric around 0)
    try:
        from matplotlib.colors import TwoSlopeNorm
        # If x/y limits are specified, compute color scaling from values within that window only
        try:
            if args.xlim is not None and len(args.xlim) == 2 and args.ylim is not None and len(args.ylim) == 2:
                xmin, xmax = float(args.xlim[0]), float(args.xlim[1])
                ymin, ymax = float(args.ylim[0]), float(args.ylim[1])
                mask_xy = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
                mask = mask_xy & np.isfinite(V2d)
                if np.any(mask):
                    local_min = float(np.min(V2d[mask]))
                    local_max = float(np.max(V2d[mask]))
                    vabs = max(abs(local_min), abs(local_max))
                else:
                    vabs = max(abs(float(np.nanmin(V2d))), abs(float(np.nanmax(V2d))))
            else:
                vabs = max(abs(float(np.nanmin(V2d))), abs(float(np.nanmax(V2d))))
        except Exception:
            vabs = max(abs(float(np.nanmin(V2d))), abs(float(np.nanmax(V2d))))
        if vabs == 0:
            vabs = 1.0
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        levels = np.linspace(-vabs, vabs, 21)
        # Use the same colormap as visualize_grid_input for consistency
        cf = ax.contourf(X, Y, V2d, levels=levels, cmap='RdBu', norm=norm)
        # Add colorbar labeled 'Value'
        cbar = plt.colorbar(cf, ax=ax, label='Value')
    except Exception as e:
        print(f"Warning: failed to plot primary GridValue values: {e}")

    # MC final snapshot set overlay: fill region where V<=0 translucently
    mc_proxy = None
    if args.mc_tag:
        try:
            mc = _load_mc_cache(args.mc_tag)
            mx = np.asarray(mc['axes'][0]); my = np.asarray(mc['axes'][1])
            MX, MY = np.meshgrid(mx, my, indexing='ij')
            snaps = mc.get('snapshots', [])
            if len(snaps) > 0:
                Zf = np.asarray(snaps[-1])
                vmin_mc = float(np.nanmin(Zf)) if np.isfinite(np.nanmin(Zf)) else 0.0
                if vmin_mc <= 0.0:
                    levels_mc = [vmin_mc, 0.0]
                    # Translucent fill
                    ax.contourf(
                        MX, MY, Zf,
                        levels=levels_mc,
                        colors=['#6A1B9A'],
                        alpha=0.28,
                        zorder=11,
                    )
                    # Solid outline at V=0
                    ax.contour(MX, MY, Zf, levels=[0.0], colors=['#6A1B9A'], linewidths=1.6, linestyles='-')
                    from matplotlib.patches import Patch
                    mc_proxy = Patch(facecolor='#6A1B9A', alpha=0.28, edgecolor='none', label='MC set (V≤0)')
                else:
                    # No V<=0 region; draw the 0-level outline solid
                    ax.contour(MX, MY, Zf, levels=[0.0], colors=['#6A1B9A'], linewidths=1.6, linestyles='-')
                    from matplotlib.lines import Line2D
                    mc_proxy = Line2D([], [], color='#6A1B9A', linestyle='-', linewidth=1.6, label='MC V=0 (no set)')
        except Exception as e:
            print(f"Warning: failed to overlay MC final snapshot for '{args.mc_tag}': {e}")

    # Baseline set overlay: match MC style — translucent fill with solid outline (deep purple)
    baseline_proxy = None
    if args.baseline_tag:
        try:
            vb = load_grid_value_by_tag(args.baseline_tag, interpolate=True)
            Xb, Yb, Vb2d, _ = _extract_slice_2d(vb, args.time, dims, slice_dim, args.slice_value)
            vmin_b = float(np.nanmin(Vb2d)) if np.isfinite(np.nanmin(Vb2d)) else 0.0
            if vmin_b <= 0.0:
                levels_b = [vmin_b, 0.0]
                cf_b = ax.contourf(
                    Xb, Yb, Vb2d,
                    levels=levels_b,
                    colors=['#6A1B9A'],
                    alpha=0.28,
                    zorder=12,
                )
                # Solid outline at V=0 for clarity (match MC outline)
                ax.contour(Xb, Yb, Vb2d, levels=[0.0], colors=['#6A1B9A'], linewidths=1.6, linestyles='-')
                from matplotlib.patches import Patch
                baseline_proxy = Patch(facecolor='#6A1B9A', alpha=0.28, edgecolor='none', label='Baseline set (V≤0)')
            else:
                # No V<=0 region; draw solid 0-level outline (match MC fallback)
                ax.contour(Xb, Yb, Vb2d, levels=[0.0], colors=['#6A1B9A'], linewidths=1.6, linestyles='-')
                from matplotlib.lines import Line2D
                baseline_proxy = Line2D([], [], color='#6A1B9A', linestyle='-', linewidth=1.6, label='Baseline V=0 (no set)')
        except Exception as e:
            print(f"Warning: failed to overlay baseline GV '{args.baseline_tag}': {e}")

    # Primary GV zero contour (solid black)
    ax.contour(X, Y, V2d, levels=[0.0], colors=['k'], linewidths=1.8)
    ax.plot([], [], color='k', linewidth=1.8, label='Robust V=0')

    # Trajectories clipped at first obstacle entry (failure_function <= 0)
    added_traj_safe_legend = False
    added_traj_collision_legend = False
    added_est_traj_legend = False
    if args.sim_tag:
        try:
            sim_path = Path('outputs') / 'simulations' / args.sim_tag / 'results.pkl'
            with open(sim_path, 'rb') as f:
                sim = pickle.load(f)
            grid_meta = sim.get('initial_states_grid_meta', None)
            # If the simulation is a grid-based run, overlay zero-level set of min-failure instead of trajectories
            if isinstance(grid_meta, dict) and grid_meta.get('type') == 'grid_slice_2d':
                # Compute min failure over time per trajectory (chunked)
                states = sim['states']  # [N, T+1, D]
                if not isinstance(states, torch.Tensor):
                    states = torch.as_tensor(states, dtype=torch.float32)
                N, T1, D = int(states.shape[0]), int(states.shape[1]), int(states.shape[2])
                sim_sys_name = sim.get('system_name', None)
                sim_system = instantiate_system_by_name(sim_sys_name) if sim_sys_name else system
                use_gpu = getattr(sim_system, '_use_gpu', False) and torch.cuda.is_available()
                device = torch.device('cuda' if use_gpu else 'cpu')
                bs = int(getattr(sim_system, '_batch_size', 100000))
                flat = states.reshape(N * T1, D)
                chunks = []
                with torch.no_grad():
                    for i0 in range(0, flat.shape[0], bs):
                        chunk = flat[i0:i0+bs].to(device)
                        vals = sim_system.failure_function(chunk, None)
                        chunks.append(vals.detach().cpu())
                vals_all = torch.cat(chunks, dim=0).reshape(N, T1)
                min_per_traj = torch.min(vals_all, dim=1).values  # [N]

                # Build grid axes
                nx, ny = int(grid_meta.get('resolution', [0, 0])[0]), int(grid_meta.get('resolution', [0, 0])[1])
                v0, v1 = int(grid_meta.get('vary_dims', [0, 1])[0]), int(grid_meta.get('vary_dims', [0, 1])[1])
                limits = grid_meta.get('limits', {})
                lo = limits.get('lo', [None, None]); hi = limits.get('hi', [None, None])
                lo0, lo1 = float(lo[0]), float(lo[1])
                hi0, hi1 = float(hi[0]), float(hi[1])

                # Ensure grid shape matches N; fallback try if mismatch
                if nx * ny != N:
                    nx = int(np.sqrt(N))
                    ny = int(np.ceil(N / max(nx, 1)))

                try:
                    Z = min_per_traj.reshape(nx, ny).numpy()
                    x = np.linspace(lo0, hi0, nx)
                    y = np.linspace(lo1, hi1, ny)
                    MX, MY = np.meshgrid(x, y, indexing='ij')
                    # Align to current plotting dims (dims)
                    i, j = dims
                    if {i, j} != {v0, v1}:
                        print(f"Warning: sim grid vary_dims=({v0},{v1}) do not match plot dims=({i},{j}); falling back to trajectories.")
                        raise RuntimeError('dims mismatch')
                    # If order differs, transpose Z or swap axes to match (i,j)
                    if (i, j) == (v0, v1):
                        XX, YY, ZZ = MX, MY, Z
                    else:
                        # (i,j) == (v1,v0): swap
                        XX, YY, ZZ = MY, MX, Z.T
                    ax.contour(XX, YY, ZZ, levels=[0.0], colors=['m'], linewidths=1.5, linestyles='-.')
                    ax.plot([], [], color='m', linestyle='-.', linewidth=1.5, label='Min-failure = 0 (sim)')
                except Exception as e:
                    print(f"Warning: failed to overlay min-failure zero level from grid sim '{args.sim_tag}': {e}")
                    # fall back to trajectories below
                    grid_meta = None
            if not (isinstance(grid_meta, dict) and grid_meta.get('type') == 'grid_slice_2d'):
                # Fallback: plot trajectories (actual only) in blue
                states = sim['states']  # [N, T+1, D]
                times = sim['times']
                est_states = sim.get('estimated_states', None)
                # Instantiate the simulation system for obstacle/failure checks
                sim_sys_name = sim.get('system_name', None)
                sim_system = instantiate_system_by_name(sim_sys_name) if sim_sys_name else system
                if not isinstance(states, torch.Tensor):
                    states = torch.as_tensor(states, dtype=torch.float32)
                if not isinstance(times, torch.Tensor):
                    times = torch.as_tensor(np.asarray(times), dtype=torch.float32)
                if est_states is not None and not isinstance(est_states, torch.Tensor):
                    est_states = torch.as_tensor(est_states, dtype=torch.float32)
                # Find first obstacle entry indices using failure_function
                hit_idx = _find_obstacle_entry_indices(sim_system or system, states)
                i, j = dims
                Sij = states[..., [i, j]].detach().cpu().numpy()  # [N, T+1, 2]
                Estij = None
                if isinstance(est_states, torch.Tensor) and est_states.numel() > 0:
                    Estij = est_states[..., [i, j]].detach().cpu().numpy()
                N, T1, _ = Sij.shape
                for n in range(N):
                    k = int(hit_idx[n].item())
                    if k == -1:
                        if args.hide_safe:
                            continue
                        # Full trajectory (actual)
                        xx = Sij[n, :, 0]; yy = Sij[n, :, 1]
                        ax.plot(xx, yy, color='tab:green', alpha=1.0, linewidth=1.8, zorder=95, label=None)
                        # Estimated trajectory (full)
                        if Estij is not None and not args.hide_est and not args.hide_safe:
                            ex = Estij[n, :, 0]; ey = Estij[n, :, 1]
                            ax.plot(ex, ey, color='tab:green', alpha=1.0, linewidth=1.2, linestyle=':', zorder=94, label=None)
                        # Start indicator
                        ax.plot(Sij[n, 0, 0], Sij[n, 0, 1], marker='o', markersize=4.0, markerfacecolor='tab:green', markeredgecolor='tab:green', markeredgewidth=0.8, linestyle='None', zorder=96)
                    else:
                        # Plot up to and including entry point (actual)
                        xx = Sij[n, :k+1, 0]; yy = Sij[n, :k+1, 1]
                        ax.plot(xx, yy, color=COLLISION_BLUE, alpha=1.0, linewidth=2.0, zorder=95, label=None)
                        # Estimated clipped similarly (align lengths)
                        if Estij is not None and not args.hide_est:
                            k_est = min(k, Estij.shape[1] - 1)
                            ex = Estij[n, :k_est+1, 0]; ey = Estij[n, :k_est+1, 1]
                            ax.plot(ex, ey, color=COLLISION_BLUE, alpha=1.0, linewidth=1.2, linestyle=':', zorder=94, label=None)
                        # Start indicator
                        ax.plot(Sij[n, 0, 0], Sij[n, 0, 1], marker='o', markersize=4.0, markerfacecolor=COLLISION_BLUE, markeredgecolor=COLLISION_BLUE, markeredgewidth=0.8, linestyle='None', zorder=96)
                        # Collision marker at entry point
                        ax.plot(Sij[n, k, 0], Sij[n, k, 1], marker='x', color=COLLISION_BLUE, markersize=6, mew=1.5, zorder=120)
                # Legend proxies
                if not args.hide_safe and not added_traj_safe_legend:
                    ax.plot([], [], color='tab:green', linewidth=1.8, label='Trajectory (safe)')
                    added_traj_safe_legend = True
                if not added_traj_collision_legend:
                    ax.plot([], [], color=COLLISION_BLUE, linewidth=2.0, label='Trajectory (collision)')
                    added_traj_collision_legend = True
                if Estij is not None and not args.hide_est and not added_est_traj_legend:
                    ax.plot([], [], color=COLLISION_BLUE, linestyle=':', linewidth=1.2, label='Trajectory (estimated)')
                    added_est_traj_legend = True
                ax.plot([], [], marker='x', color=COLLISION_BLUE, linestyle='None', label='Obstacle entry (failure<=0)')
        except Exception as e:
            print(f"Warning: failed to overlay trajectories for '{args.sim_tag}': {e}")
    # Secondary trajectories overlay from a second simulation tag (always treated as trajectories)
    if args.sim_tag2:
        try:
            sim2_path = Path('outputs') / 'simulations' / args.sim_tag2 / 'results.pkl'
            with open(sim2_path, 'rb') as f:
                sim2 = pickle.load(f)
            states2 = sim2['states']
            times2 = sim2['times']
            est_states2 = sim2.get('estimated_states', None)
            sim2_sys_name = sim2.get('system_name', None)
            sim2_system = instantiate_system_by_name(sim2_sys_name) if sim2_sys_name else system
            if not isinstance(states2, torch.Tensor):
                states2 = torch.as_tensor(states2, dtype=torch.float32)
            if not isinstance(times2, torch.Tensor):
                times2 = torch.as_tensor(np.asarray(times2), dtype=torch.float32)
            if est_states2 is not None and not isinstance(est_states2, torch.Tensor):
                est_states2 = torch.as_tensor(est_states2, dtype=torch.float32)
            # Find first obstacle entry indices using failure_function
            hit2 = _find_obstacle_entry_indices(sim2_system or system, states2)
            i, j = dims
            Sij2 = states2[..., [i, j]].detach().cpu().numpy()
            Estij2 = None
            if isinstance(est_states2, torch.Tensor) and est_states2.numel() > 0:
                Estij2 = est_states2[..., [i, j]].detach().cpu().numpy()
            N2, T12, _ = Sij2.shape
            for n in range(N2):
                k = int(hit2[n].item())
                if k == -1:
                    if not args.hide_safe:
                        xx = Sij2[n, :, 0]; yy = Sij2[n, :, 1]
                        ax.plot(xx, yy, color='tab:green', alpha=1.0, linewidth=1.6, zorder=95, label=None)
                        if Estij2 is not None and not args.hide_est:
                            ex = Estij2[n, :, 0]; ey = Estij2[n, :, 1]
                            ax.plot(ex, ey, color='tab:green', alpha=1.0, linewidth=1.1, linestyle=':', zorder=94, label=None)
                        # Start indicator
                        ax.plot(Sij2[n, 0, 0], Sij2[n, 0, 1], marker='o', markersize=3.8, markerfacecolor='tab:green', markeredgecolor='tab:green', markeredgewidth=0.8, linestyle='None', zorder=96)
                else:
                    xx = Sij2[n, :k+1, 0]; yy = Sij2[n, :k+1, 1]
                    ax.plot(xx, yy, color=COLLISION_BLUE, alpha=1.0, linewidth=1.8, zorder=95, label=None)
                    if Estij2 is not None and not args.hide_est:
                        k_est = min(k, Estij2.shape[1] - 1)
                        ex = Estij2[n, :k_est+1, 0]; ey = Estij2[n, :k_est+1, 1]
                        ax.plot(ex, ey, color=COLLISION_BLUE, alpha=1.0, linewidth=1.1, linestyle=':', zorder=94, label=None)
                    # Start indicator
                    ax.plot(Sij2[n, 0, 0], Sij2[n, 0, 1], marker='o', markersize=3.8, markerfacecolor=COLLISION_BLUE, markeredgecolor=COLLISION_BLUE, markeredgewidth=0.8, linestyle='None', zorder=96)
                    ax.plot(Sij2[n, k, 0], Sij2[n, k, 1], marker='x', color=COLLISION_BLUE, markersize=6, mew=1.5, zorder=120)
            # No additional legend entries; uses the same 'Trajectory' proxy
        except Exception as e:
            print(f"Warning: failed to overlay trajectories for second sim '{args.sim_tag2}': {e}")

    # Labels and formatting
    if system is not None:
        xi, yj = _axis_labels(system, dims)
    else:
        i, j = dims
        xi, yj = f'x{i}', f'x{j}'
    ax.set_xlabel(xi); ax.set_ylabel(yj)
    title_bits = [f"Value: {args.value_tag}"]
    if args.baseline_tag: title_bits.append(f"Baseline: {args.baseline_tag}")
    if args.mc_tag: title_bits.append(f"MC: {args.mc_tag}")
    if args.sim_tag: title_bits.append(f"Sim: {args.sim_tag}")
    if args.sim_tag2: title_bits.append(f"Sim2: {args.sim_tag2}")
    ax.set_title(" | ".join(title_bits))
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    # Optional zoom limits
    if args.xlim is not None and len(args.xlim) == 2:
        try:
            ax.set_xlim(float(args.xlim[0]), float(args.xlim[1]))
        except Exception:
            pass
    if args.ylim is not None and len(args.ylim) == 2:
        try:
            ax.set_ylim(float(args.ylim[0]), float(args.ylim[1]))
        except Exception:
            pass

    # Obstacles on (x,y) plane only
    if system is not None and set(dims) == {0, 1}:
        _plot_obstacles_if_xy(ax, system)

    # Place legend outside the plotting area on the right (within reserved figure space)
    # Use multiple columns when many legend entries exist to reduce width
    try:
        handles, labels = ax.get_legend_handles_labels()
        # Inject proxy handle for baseline set if we created one
        if baseline_proxy is not None:
            handles.append(baseline_proxy)
            labels.append(baseline_proxy.get_label())
        # Inject MC proxy if created
        if 'mc_proxy' in locals() and mc_proxy is not None:
            handles.append(mc_proxy)
            labels.append(mc_proxy.get_label())
        n_items = len(labels)
    except Exception:
        handles, labels, n_items = [], [], 0
    ncol = 1 if n_items <= 10 else (2 if n_items <= 20 else 3)
    legend = ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True, ncol=ncol)

    # Save or show
    if args.interactive:
        # Do not create save directories in interactive mode
        plt.show()
    else:
        if args.save_dir is None:
            parts = [f"val_{args.value_tag}"]
            if args.baseline_tag: parts.append(f"base_{args.baseline_tag}")
            if args.mc_tag: parts.append(f"mc_{args.mc_tag}")
            if args.sim_tag: parts.append(f"sim_{args.sim_tag}")
            out_dir = Path('outputs') / 'visualizations' / 'value_evaluation' / "__".join(parts)
        else:
            out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'comparison.png'
        fig.tight_layout()
        # Save with padding; legend is within figure due to subplots_adjust
        fig.savefig(out_path, dpi=int(args.dpi), bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        print(f"\n✓ Saved comparison: {out_path}")


if __name__ == '__main__':
    main()
