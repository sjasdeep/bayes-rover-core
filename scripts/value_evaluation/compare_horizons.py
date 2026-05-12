#!/usr/bin/env python3
"""
Compare t=0 reachable sets (V<=0) across multiple GridValues built with different time horizons.

This script overlays the t=0 set from each provided GridValue tag on a common 2D slice.
Useful for visualizing how the initial back-reachable set changes with horizon length.

Examples:
  # Compare three horizons of the same dynamics/input setup
  python scripts/value_evaluation/compare_horizons.py \
    --value-tags brt_H2 brt_H4 brt_H6 \
    --slice-dim 2 --slice-value 0.0 \
    --dpi 180

Notes:
- To ensure fair comparison across tags with possibly different grids, this script evaluates
  each GridValue at t=0 on a common reference grid (taken from the first tag's axes) using
  the GridValue.value(x, t) query function for interpolation.
- For state_dim > 2, provide --slice-dim and optionally --slice-value; dims to plot default
  to the first two dims excluding slice-dim, or specify with --dims.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impl.values.grid_value import GridValue
from src.utils.cache_loaders import (  # type: ignore
    load_grid_value_by_tag,
    get_grid_value_metadata,
    instantiate_system_by_name,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--value-tags', nargs='+', required=True, help='List of GridValue cache tags to compare (built with different horizons)')
    ap.add_argument('--value-tags2', nargs='+', default=None, help='Optional second list of GridValue tags to compare against (rendered dashed)')
    ap.add_argument('--dims', type=int, nargs=2, default=None, help='State dims to plot (default: first two excluding slice-dim when state_dim>2, else [0,1])')
    ap.add_argument('--slice-dim', type=int, default=None, help='Slice dimension for state_dim>2 (default: first dim not in --dims)')
    ap.add_argument('--slice-value', type=float, default=None, help='Slice value for slice-dim (default: middle of axis)')
    ap.add_argument('--fill', action='store_true', help='Fill each set translucently in addition to the outline')
    ap.add_argument('--alpha', type=float, default=0.28, help='Alpha for filled sets when --fill is used (default: 0.28)')
    ap.add_argument('--cmap', type=str, default='RdBu', help='Matplotlib colormap for ordering by horizon (default: RdBu; matches compare_values theme)')
    ap.add_argument('--cmap-start-second', action='store_true', help='Bias color mapping so the smallest horizon uses the second-step color (skips the lightest tone).')
    ap.add_argument('--cmap-gap', type=float, default=0.12, help='Exclude a central band of the colormap around the midpoint (0 disables). Useful for diverging maps like RdBu where the center is near-white and hard to see. Range: [0, 0.49].')
    ap.add_argument('--labels', nargs='*', default=None, help='Optional labels (same length as value-tags); defaults to tag (H=metadata)')
    ap.add_argument('--labels2', nargs='*', default=None, help='Optional labels for --value-tags2 (same length as value-tags2)')
    ap.add_argument('--cmap2', type=str, default='plasma', help='Colormap for --value-tags2 (default: plasma)')
    ap.add_argument('--fill2', action='store_true', help='Fill second set translucently too (optional)')
    ap.add_argument('--palette', type=str, choices=['bright'], default=None, help="Preset discrete palette for horizons (e.g., 'bright' = highly distinct 5-color set)")
    ap.add_argument('--colors', nargs='+', default=None, help='Explicit list of colors (hex or names) to map to sorted horizons (H1..HK)')
    ap.add_argument('--mono-color', type=str, default=None, help='Use a single color for all contours across both sets (e.g., red)')
    ap.add_argument('--vary-lw', action='store_true', help='Vary line width from lw-min to lw-max across horizons (small->large)')
    ap.add_argument('--lw-min', type=float, default=1.0, help='Minimum line width for smallest horizon when --vary-lw is used')
    ap.add_argument('--lw-max', type=float, default=4.0, help='Maximum line width for largest horizon when --vary-lw is used')
    ap.add_argument('--lw', type=float, default=1.2, help='Line width for contour outlines (default: 1.2)')
    ap.add_argument('--save-dir', type=str, default=None, help='Directory to save figure (default under outputs/visualizations/value_horizon_compare)')
    ap.add_argument('--dpi', type=int, default=150, help='Figure DPI for saved image (default: 150)')
    ap.add_argument('--interactive', action='store_true', help='Open interactive window instead of saving (if backend available)')
    ap.add_argument('--xlim', type=float, nargs=2, default=None, help='x-axis limits [xmin xmax]')
    ap.add_argument('--ylim', type=float, nargs=2, default=None, help='y-axis limits [ymin ymax]')
    return ap.parse_args()


def _pick_dims_and_slice(vf: GridValue, dims_arg: Optional[Tuple[int,int]], slice_dim_arg: Optional[int], slice_value: Optional[float]):
    D = vf.state_dim
    if D == 2:
        return (0, 1), None, None
    if dims_arg is not None:
        dims = (int(dims_arg[0]), int(dims_arg[1]))
    else:
        # choose first two dims; adjust below if a slice-dim is specified and conflicts
        dims = (0, 1)
    all_dims = list(range(D))
    if slice_dim_arg is None:
        rest = [d for d in all_dims if d not in dims]
        if not rest:
            # pick next available dim cyclically
            slice_dim = (set(all_dims) - set(dims)).pop() if set(all_dims) - set(dims) else 0
        else:
            slice_dim = int(rest[0])
    else:
        slice_dim = int(slice_dim_arg)
    # Ensure slice_dim not in dims
    if slice_dim in dims:
        rest = [d for d in all_dims if d != slice_dim]
        dims = (rest[0], rest[1])
    return dims, slice_dim, slice_value


def _build_reference_grid(vf: GridValue, dims: Tuple[int,int], slice_dim: Optional[int], slice_value: Optional[float]):
    i, j = int(dims[0]), int(dims[1])
    # Axes from the first GridValue
    Xi = vf._axes[i].detach().cpu().numpy()
    Yj = vf._axes[j].detach().cpu().numpy()
    X, Y = np.meshgrid(Xi, Yj, indexing='ij')
    D = vf.state_dim
    # Default other dims to middle values
    mids = []
    for d in range(D):
        axd = vf._axes[d]
        mids.append(float(axd[int(axd.numel()//2)].item()))
    state_grid = np.zeros((X.size, D), dtype=np.float32)
    for d in range(D):
        state_grid[:, d] = mids[d]
    state_grid[:, i] = X.reshape(-1)
    state_grid[:, j] = Y.reshape(-1)
    if slice_dim is not None:
        if slice_value is None:
            # middle value
            axs = vf._axes[slice_dim]
            sv = float(axs[int(axs.numel()//2)].item())
        else:
            sv = float(slice_value)
        state_grid[:, slice_dim] = sv
    return X, Y, state_grid


def _plot_obstacles_if_xy(ax, system):
    """Draw 2D obstacles on (x,y) plane using shared utility."""
    from src.utils.obstacles import draw_obstacles_2d
    try:
        draw_obstacles_2d(ax, system)
    except Exception:
        pass


def _extract_horizon(meta: dict, tag: str) -> Optional[float]:
    """Try to obtain the time horizon from metadata; fallback to parsing `_H{number}` in the tag.
    Returns None if not found.
    """
    H = meta.get('time_horizon') or meta.get('metadata', {}).get('time_horizon')
    if H is not None:
        try:
            return float(H)
        except Exception:
            pass
    # Fallback parse from tag like ..._H2, _H2.5, etc.
    try:
        m = re.search(r"_H(\d+(?:\.\d+)?)", tag)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def main():
    args = parse_args()

    if args.interactive:
        try:
            matplotlib.use('TkAgg', force=True)
        except Exception:
            pass
        try:
            import matplotlib as mpl
            mpl.rcParams['figure.dpi'] = int(args.dpi)
        except Exception:
            pass

    # Load first GV to determine dims and reference grid
    tags: List[str] = list(args.value_tags)
    vfs: List[GridValue] = []
    metas = []
    for t in tags:
        vf = load_grid_value_by_tag(t, interpolate=True)
        vfs.append(vf)
        metas.append(get_grid_value_metadata(t))

    # Optional: sort by horizon (ascending)
    try:
        order = sorted(range(len(tags)), key=lambda k: float(metas[k].get('time_horizon', metas[k].get('metadata', {}).get('time_horizon', 0.0))))
        tags = [tags[k] for k in order]
        vfs = [vfs[k] for k in order]
        metas = [metas[k] for k in order]
    except Exception:
        pass

    vf0 = vfs[0]
    dims, slice_dim, slice_value = _pick_dims_and_slice(vf0, tuple(args.dims) if args.dims else None, args.slice_dim, args.slice_value)
    X, Y, states_ref = _build_reference_grid(vf0, dims, slice_dim, slice_value)

    # Prepare plot
    fig, ax = plt.subplots(figsize=(8, 7))
    try:
        fig.subplots_adjust(right=0.68)
    except Exception:
        pass

    # Colors per horizon using a common colormap for both sets to ensure matching hues per H
    cmap = plt.get_cmap(args.cmap)
    cmap2 = None  # use common mapping for both sets when present
    # Build values per tag at t=0 on the reference grid
    T = states_ref.shape[0]
    times0 = torch.zeros((T,), dtype=torch.float32)
    states_t = torch.from_numpy(states_ref).to(torch.float32)

    labels = args.labels if args.labels and len(args.labels) == len(tags) else None

    # Build a unified, ordered list of unique horizons across both sets to map colors smoothly
    H_list: List[Optional[float]] = []
    H_list += [_extract_horizon(m, t) for t, m in zip(tags, metas)]
    if getattr(args, 'value_tags2', None):
        # We'll populate metas2 later; collect tags2 here as placeholders
        pass
    # We'll compute full union once we load tags2 (if any)

    # Prepare optional second set first to compute union of horizons for consistent coloring
    tags2: List[str] = []
    vfs2: List[GridValue] = []
    metas2: List[dict] = []
    if getattr(args, 'value_tags2', None):
        tags2 = list(args.value_tags2)
        for t2 in tags2:
            vf2 = load_grid_value_by_tag(t2, interpolate=True)
            vfs2.append(vf2)
            metas2.append(get_grid_value_metadata(t2))
        # Sort by horizon if available (for nicer legend ordering); colors will use union mapping
        try:
            order2 = sorted(range(len(tags2)), key=lambda k: float(_extract_horizon(metas2[k], tags2[k]) or 0.0))
            tags2 = [tags2[k] for k in order2]
            vfs2 = [vfs2[k] for k in order2]
            metas2 = [metas2[k] for k in order2]
        except Exception:
            pass

    # Build union of horizons and map to colors
    def _h_key(h: Optional[float]) -> Optional[float]:
        return None if h is None else float(h)

    H_all: List[float] = []
    for t, m in zip(tags, metas):
        h = _extract_horizon(m, t)
        if h is not None:
            H_all.append(h)
    for t2, m2 in zip(tags2, metas2):
        h2 = _extract_horizon(m2, t2)
        if h2 is not None:
            H_all.append(h2)
    H_sorted = sorted(set([round(h, 6) for h in H_all]))
    h_to_idx: Dict[float, int] = {h: i for i, h in enumerate(H_sorted)}
    denom = max(1, len(H_sorted) - 1)

    # Optional discrete palettes or explicit colors
    BRIGHT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    explicit_colors = None
    if getattr(args, 'colors', None):
        explicit_colors = list(args.colors)
    elif getattr(args, 'palette', None) == 'bright':
        explicit_colors = BRIGHT_COLORS[:max(1, len(H_sorted))]

    def color_for(meta: dict, tag: str, fallback_idx: int, total: int):
        # Mono color overrides everything
        if getattr(args, 'mono_color', None):
            return args.mono_color
        h = _extract_horizon(meta, tag)
        if h is None:
            # Fallback if unknown horizon
            if explicit_colors is not None and len(explicit_colors) > 0:
                return explicit_colors[fallback_idx % len(explicit_colors)]
            pos_base = float(fallback_idx) / max(1, total - 1)
            # Apply start bias if requested
            if getattr(args, 'cmap_start_second', False):
                low = (1.0 / denom) if denom > 0 else 0.0
                pos_base = low + (1.0 - low) * pos_base
            return cmap(pos_base)
        key = round(h, 6)
        if explicit_colors is not None and len(explicit_colors) > 0:
            idx = h_to_idx.get(key, 0)
            # If not enough colors provided, cycle
            return explicit_colors[idx % len(explicit_colors)]
        pos = float(h_to_idx.get(key, 0)) / denom if denom > 0 else 0.0
        if getattr(args, 'cmap_start_second', False):
            low = (1.0 / denom) if denom > 0 else 0.0
            pos = low + (1.0 - low) * pos
        return cmap(pos)

    def lw_for(meta: dict, tag: str, fallback_idx: int, total: int) -> float:
        if not getattr(args, 'vary_lw', False):
            return float(args.lw)
        lw_min = float(args.lw_min)
        lw_max = float(args.lw_max)
        if len(H_sorted) <= 1:
            return lw_max
        h = _extract_horizon(meta, tag)
        if h is None:
            # Map by fallback sequence
            frac = float(fallback_idx) / max(1, total - 1)
            return lw_min + frac * (lw_max - lw_min)
        idx_h = h_to_idx.get(round(h, 6), 0)
        frac = float(idx_h) / max(1, len(H_sorted) - 1)
        return lw_min + frac * (lw_max - lw_min)

    def z_for(meta: dict, tag: str, fallback_idx: int, total: int, delta: float = 0.0) -> float:
        """Higher z for smaller horizons so H1 is on top. delta allows slight offset between sets."""
        if len(H_sorted) > 0:
            h = _extract_horizon(meta, tag)
            if h is not None:
                idx_h = h_to_idx.get(round(h, 6), 0)
                # reverse so smallest (idx 0) gets highest z
                return 20.0 + float(len(H_sorted) - 1 - idx_h) + float(delta)
        # Fallback to sequence position
        return 20.0 + float(max(0, total - 1 - fallback_idx)) + float(delta)

    for idx, (tag, vf, meta) in enumerate(zip(tags, vfs, metas)):
        # Use normalized index for color
        c = color_for(meta, tag, idx, len(tags))
        with torch.no_grad():
            V = vf.value(states_t, times0).reshape(X.shape[0], X.shape[1])
        Vn = V.detach().cpu().numpy()
        # Determine fill region
        try:
            vmin = float(np.nanmin(Vn)) if np.isfinite(np.nanmin(Vn)) else 0.0
        except Exception:
            vmin = 0.0
        # Outline at V=0 always
        z = z_for(meta, tag, idx, len(tags), delta=0.0)
        lw = lw_for(meta, tag, idx, len(tags))
        cs = ax.contour(X, Y, Vn, levels=[0.0], colors=[c], linewidths=lw, linestyles='-', zorder=z)
        # Add a thin dark stroke so light colors (e.g., near-white center of RdBu) remain visible
        try:
            for coll in cs.collections:
                coll.set_path_effects([pe.Stroke(linewidth=max(lw + 0.8, lw * 1.6), foreground='k'), pe.Normal()])
        except Exception:
            pass
        if args.fill and vmin <= 0.0:
            ax.contourf(X, Y, Vn, levels=[vmin, 0.0], colors=[c], alpha=float(args.alpha), zorder=11)
        # Legend label
        H = meta.get('time_horizon') or meta.get('metadata', {}).get('time_horizon')
        lbl = (labels[idx] if labels is not None else f"{tag} (H={float(H):.2f}s)" if H is not None else tag)
        ax.plot([], [], color=c, linewidth=lw_for(meta, tag, idx, len(tags)), label=lbl)

    # Optional second set: dashed outlines (and optional fill) with shared color mapping
    if getattr(args, 'value_tags2', None):
        labels2 = args.labels2 if args.labels2 and len(args.labels2) == len(tags2) else None

        for idx2, (tag2, vf2, meta2) in enumerate(zip(tags2, vfs2, metas2)):
            # Ensure compatible state dimension
            if vf2.state_dim != vf0.state_dim:
                print(f"[warn] Skipping tag '{tag2}' due to mismatched state_dim ({vf2.state_dim} != {vf0.state_dim})")
                continue
            c2 = color_for(meta2, tag2, idx2, len(tags2))
            with torch.no_grad():
                V2 = vf2.value(states_t, times0).reshape(X.shape[0], X.shape[1])
            V2n = V2.detach().cpu().numpy()
            # Determine fill region for second set if requested
            try:
                vmin2 = float(np.nanmin(V2n)) if np.isfinite(np.nanmin(V2n)) else 0.0
            except Exception:
                vmin2 = 0.0
            # Dashed outline at V=0
            z2 = z_for(meta2, tag2, idx2, len(tags2), delta=0.15)
            lw2 = lw_for(meta2, tag2, idx2, len(tags2))
            cs2 = ax.contour(X, Y, V2n, levels=[0.0], colors=[c2], linewidths=lw2, linestyles='--', zorder=z2)
            try:
                for coll in cs2.collections:
                    coll.set_path_effects([pe.Stroke(linewidth=max(lw2 + 0.7, lw2 * 1.5), foreground='k'), pe.Normal()])
            except Exception:
                pass
            if getattr(args, 'fill2', False) and vmin2 <= 0.0:
                ax.contourf(X, Y, V2n, levels=[vmin2, 0.0], colors=[c2], alpha=float(args.alpha), zorder=10)
            # Legend label for second set
            H2 = meta2.get('time_horizon') or meta2.get('metadata', {}).get('time_horizon')
            lbl2 = (labels2[idx2] if labels2 is not None else f"{tag2} (H={float(H2):.2f}s)" if H2 is not None else tag2)
            ax.plot([], [], color=c2, linewidth=lw_for(meta2, tag2, idx2, len(tags2)), linestyle='--', label=lbl2)

    # Axes labels from system and optional obstacle overlay
    try:
        sys_name = metas[0].get('system_name', metas[0].get('system', ''))
        system = instantiate_system_by_name(sys_name) if sys_name else None
        if system is not None:
            labels = getattr(system, 'state_labels', tuple([f'x{i}' for i in range(vf0.state_dim)]))
            i, j = dims
            ax.set_xlabel(labels[i] if i < len(labels) else f'x{i}')
            ax.set_ylabel(labels[j] if j < len(labels) else f'x{j}')
            # Plot obstacles only when viewing the (x,y) plane
            try:
                if set(dims) == {0, 1}:
                    _plot_obstacles_if_xy(ax, system)
            except Exception:
                pass
    except Exception:
        pass

    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
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

    try:
        handles, labels = ax.get_legend_handles_labels()
        n_items = len(labels)
    except Exception:
        handles, labels, n_items = [], [], 0
    ncol = 1 if n_items <= 10 else (2 if n_items <= 20 else 3)
    ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True, ncol=ncol)

    if args.interactive:
        plt.show()
    else:
        out_dir = Path(args.save_dir) if args.save_dir else (Path('outputs') / 'visualizations' / 'value_horizon_compare')
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = "__".join(tags)
        if getattr(args, 'value_tags2', None):
            safe2 = "__".join(list(args.value_tags2))
            fname = f"compare_horizons__{safe}__vs__{safe2}.png"
        else:
            fname = f"compare_horizons__{safe}.png"
        out_path = out_dir / fname
        fig.tight_layout()
        fig.savefig(out_path, dpi=int(args.dpi), bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        print(f"\n✓ Saved horizon comparison: {out_path}")


if __name__ == '__main__':
    main()
