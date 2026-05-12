#!/usr/bin/env python3
"""
Visualize Monte Carlo value caches.

Default: overlay zero-level contours for snapshots. Options allow overlaying the
final zero contour and/or plotting the full final value function as a filled
contour with a colorbar (similar to GridValue visualization).

Usage:
    python scripts/monte_carlo_value/visualize.py --tag my_mc
    python scripts/monte_carlo_value/visualize.py --tag my_mc --show-final
    python scripts/monte_carlo_value/visualize.py --tag my_mc --show-final-field
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')  # default to non-interactive; switch with --interactive
import numpy as np
import torch
from src.utils.cache_loaders import instantiate_system_by_name


def load_cache(tag: str):
    path = Path('.cache') / 'monte_carlo_values' / f'{tag}.pkl'
    if not path.exists():
        raise SystemExit(f"Cache not found: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Ensure the tag is present for naming outputs
    if isinstance(data, dict) and 'tag' not in data:
        data['tag'] = tag
    return data


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def plot_zero_contours(
    data,
    *,
    show_final: bool,
    show_final_field: bool = False,
    interactive: bool = False,
    save: bool = True,
):
    axes = data['axes']
    x = _to_numpy(axes[0])
    y = _to_numpy(axes[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    snapshots = data.get('snapshots', [])
    V = data.get('value')
    meta = data.get('meta', {})

    if interactive:
        try:
            matplotlib.use('TkAgg', force=True)
        except Exception:
            pass
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    # Dynamic title based on mode
    title_bits = ["Monte Carlo"]
    if show_final_field:
        title_bits.append("final value field")
    else:
        title_bits.append("zero contours")
    sys_name = data.get('system_name')
    ctrl_name = data.get('control_name')
    if sys_name or ctrl_name:
        title_bits.append(f"({sys_name} / {ctrl_name})")
    ax.set_title(" ".join(title_bits))
    # Derive axis labels from system state_labels and vary_dims (if available)
    state_labels = list(data.get('state_labels', ['x', 'y']))
    vary_dims = meta.get('vary_dims', [0, 1])
    def _label_for(idx: int) -> str:
        try:
            return state_labels[int(idx)]
        except Exception:
            return ['x', 'y'][idx]
    ax.set_xlabel(_label_for(vary_dims[0] if len(vary_dims) > 0 else 0))
    ax.set_ylabel(_label_for(vary_dims[1] if len(vary_dims) > 1 else 1))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Optionally plot the full final value field as a filled contour
    if show_final_field and V is not None:
        Vnp = _to_numpy(V)
        from matplotlib.colors import TwoSlopeNorm
        vabs = float(max(abs(np.nanmin(Vnp)), abs(np.nanmax(Vnp)))) if Vnp.size else 1.0
        vabs = vabs if vabs > 0 else 1.0
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        levels = np.linspace(-vabs, vabs, 21)
        cf = ax.contourf(X, Y, Vnp, levels=levels, cmap='RdYlBu', norm=norm)
        # Shared colorbar labeled as Value
        sm = plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu')
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Value')
        # Always overlay zero contour for clarity
        ax.contour(X, Y, Vnp, levels=[0.0], colors=['k'], linewidths=2.0)
        # Legend proxy for final zero contour
        ax.plot([], [], color='k', linewidth=2.0, label='Final (V=0)')

    # Draw snapshot zero contours with progressively darker colors
    n = len(snapshots)
    for i, Z in enumerate(snapshots):
        Znp = _to_numpy(Z)
        color = plt.cm.Blues(0.3 + 0.7 * (i + 1) / max(1, n))
        # Draw contour
        ax.contour(X, Y, Znp, levels=[0.0], colors=[color], linewidths=1.5)
        # Add a proxy handle for legend since some Matplotlib versions don't expose collections
        ax.plot([], [], color=color, linewidth=1.5, label=f"N={i+1} snap")

    if show_final and V is not None:
        Vnp = _to_numpy(V)
        ax.contour(X, Y, Vnp, levels=[0.0], colors=['k'], linewidths=2.0)
        ax.plot([], [], color='k', linewidth=2.0, label='Final')

    # Optionally overlay obstacles when the slice corresponds to the (x, y) plane
    try:
        sys_name = data.get('system_name', None)
        vary_dims = meta.get('vary_dims', [0, 1])
        # Only render obstacles when the plotted plane is exactly x-y (order-insensitive)
        if sys_name is not None and set(int(d) for d in vary_dims) == {0, 1}:
            system = instantiate_system_by_name(sys_name)
            from src.utils.obstacles import draw_obstacles_2d
            draw_obstacles_2d(ax, system, zorder=10)
    except Exception as e:
        # Keep visualization robust if system instantiation or obstacle plotting fails
        print(f"Warning: obstacle rendering skipped ({e})")

    ax.legend(loc='upper right')
    fig.tight_layout()

    # Default behavior: save unless explicitly disabled via --no-save (parsed default=True)

    if save:
        out_dir = Path('outputs') / 'visualizations' / 'monte_carlo'
        out_dir.mkdir(parents=True, exist_ok=True)
        # Prefer tag-based filename for clarity; fallback to system/control
        tag = data.get('tag') or f"{data.get('system_name')}_{data.get('control_name')}"
        suffix = '_field' if show_final_field else ''
        out_path = out_dir / f"{tag}{suffix}.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved visualization: {out_path}")

    if interactive:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', type=str, required=True)
    ap.add_argument('--interactive', action='store_true', help='Show an interactive window (TkAgg). If omitted, no window is shown.')
    ap.add_argument('--show-final', action='store_true', help='Overlay final V_N=0 contour in black')
    ap.add_argument('--show-final-field', action='store_true',
                    help='Plot the full final value function as a filled contour with colorbar and overlay its zero contour')
    # Default to saving; allow disabling with --no-save only
    ap.add_argument('--no-save', dest='save', action='store_false', help='Do not save an image file', default=True)
    args = ap.parse_args()

    data = load_cache(args.tag)
    plot_zero_contours(
        data,
        show_final=args.show_final,
        show_final_field=args.show_final_field,
        interactive=args.interactive,
        save=args.save,
    )
