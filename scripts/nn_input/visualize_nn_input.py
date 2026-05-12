#!/usr/bin/env python3
"""
Visualize NNInput from a tagged cache.

Usage:
  python scripts/nn_input/visualize_nn_input.py --tag TAG [--preset PRESET] [--save-dir DIR]
  python scripts/nn_input/visualize_nn_input.py --tag TAG --interactive  # Interactive mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_visualization_presets
from src.utils.cache_loaders import (
    get_nn_input_metadata,
    instantiate_system_by_name,
    load_nn_input_by_tag,
)
from src.utils.cache_loaders import load_grid_input_by_tag
from src.utils.interactive_viz import InteractiveVisualizer, SliderSpec, create_time_slider


def visualize_2d_slice(nn_input, system, time_val, dim1=0, dim2=1, fixed_dims=None, vis_resolution=None, ax=None, fig=None, output_dim=None, ref_grid=None):
    """
    Visualize a 2D slice by sampling the NNInput over a grid on (dim1, dim2).
    """
    if system.state_dim < 2:
        print("System has less than 2 dimensions, cannot create 2D visualization")
        return

    fixed_dims = dict(fixed_dims or {})
    for i in range(system.state_dim):
        if i not in [dim1, dim2] and i not in fixed_dims:
            lower = system.state_limits[0, i].item()
            upper = system.state_limits[1, i].item()
            if torch.isinf(torch.tensor(lower)) or torch.isinf(torch.tensor(upper)):
                raise ValueError(f"Cannot visualize: dimension {i} has infinite limits")
            fixed_dims[i] = (lower + upper) / 2

    # Sampling resolution
    if vis_resolution is None:
        # Default visualization resolution per plotted dimension
        resolution1 = 100
        resolution2 = 100
        lower1 = system.state_limits[0, dim1].item()
        upper1 = system.state_limits[1, dim1].item()
        lower2 = system.state_limits[0, dim2].item()
        upper2 = system.state_limits[1, dim2].item()
        vals1 = torch.linspace(lower1, upper1, resolution1)
        vals2 = torch.linspace(lower2, upper2, resolution2)
    else:
        resolution1 = int(vis_resolution[dim1])
        resolution2 = int(vis_resolution[dim2])
        lower1 = system.state_limits[0, dim1].item()
        upper1 = system.state_limits[1, dim1].item()
        lower2 = system.state_limits[0, dim2].item()
        upper2 = system.state_limits[1, dim2].item()
        vals1 = torch.linspace(lower1, upper1, resolution1)
        vals2 = torch.linspace(lower2, upper2, resolution2)

    V1, V2 = torch.meshgrid(vals1, vals2, indexing='ij')

    states = torch.zeros(resolution1, resolution2, system.state_dim)
    states[:, :, dim1] = V1
    states[:, :, dim2] = V2
    for dim, val in fixed_dims.items():
        states[:, :, dim] = float(val)

    flat_states = states.reshape(-1, system.state_dim)

    # Evaluate NNInput
    outputs = nn_input.input(flat_states, float(time_val))
    output_grid = outputs.reshape(resolution1, resolution2, -1)

    # Optional: evaluate reference GridInput on the same points for comparison
    truth_grid = None
    if ref_grid is not None:
        with torch.no_grad():
            truth = ref_grid.input(flat_states, float(time_val))
        truth_grid = truth.reshape(resolution1, resolution2, -1)

    # Limits for consistent color scales based on input role
    role = getattr(nn_input, 'type', 'control')
    if role == 'control':
        lower, upper = system.control_limits(flat_states, float(time_val))
    elif role == 'disturbance':
        lower, upper = system.disturbance_limits(flat_states, float(time_val))
    else:
        lower, upper = system.uncertainty_limits(flat_states, float(time_val))
    ctrl_lower = lower.reshape(resolution1, resolution2, -1)
    ctrl_upper = upper.reshape(resolution1, resolution2, -1)

    num_outputs = output_grid.shape[-1]

    if ax is None:
        # If we have a truth grid, default to a 3-panel comparison (Truth, Prediction, Error) for a single output
        if (truth_grid is not None) and (num_outputs == 1):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            compare_mode = True
        else:
            compare_mode = False
            if fig is None:
                fig, axes = plt.subplots(1, num_outputs, figsize=(6*num_outputs, 5))
            else:
                axes = fig.subplots(1, num_outputs)
            if num_outputs == 1:
                axes = [axes]
    else:
        if output_dim is not None:
            axes = [ax]
            output_grid = output_grid[:, :, output_dim:output_dim+1]
            ctrl_lower = ctrl_lower[:, :, output_dim:output_dim+1]
            ctrl_upper = ctrl_upper[:, :, output_dim:output_dim+1]
        else:
            axes = [ax]
            output_grid = output_grid[:, :, 0:1]
            ctrl_lower = ctrl_lower[:, :, 0:1]
            ctrl_upper = ctrl_upper[:, :, 0:1]
        if fig is None:
            fig = ax.figure

    dim1_label = system.state_labels[dim1] if dim1 < len(system.state_labels) else f"dim_{dim1}"
    dim2_label = system.state_labels[dim2] if dim2 < len(system.state_labels) else f"dim_{dim2}"

    if ax is None and (truth_grid is not None) and (num_outputs == 1) and compare_mode:
        # Comparison panels
        out_idx = output_dim if output_dim is not None else 0
        Zp = output_grid[:, :, out_idx].detach().cpu().numpy()
        Zt = truth_grid[:, :, out_idx].detach().cpu().numpy()
        # Shared scale for truth/pred using control limits for consistency
        vmin = float(ctrl_lower[:, :, out_idx].min().item())
        vmax = float(ctrl_upper[:, :, out_idx].max().item())
        # Plot
        c0 = axes[0].contourf(V1.numpy(), V2.numpy(), Zt, levels=20, cmap='RdBu', vmin=vmin, vmax=vmax)
        c1 = axes[1].contourf(V1.numpy(), V2.numpy(), Zp, levels=20, cmap='RdBu', vmin=vmin, vmax=vmax)
        E = (Zp - Zt)
        # Error: symmetric bounds based on control-limit magnitude (consistent with trainer)
        emax = float(max(abs(vmin), abs(vmax))) or 1.0
        c2 = axes[2].contourf(V1.numpy(), V2.numpy(), E, levels=20, cmap='coolwarm', vmin=-emax, vmax=emax)
        # Labels
        for a, title in zip(axes, ['Truth', 'Prediction', 'Error (pred - truth)']):
            a.set_xlabel(dim1_label)
            a.set_ylabel(dim2_label)
            a.set_title(title)
            a.grid(True, alpha=0.3)
        fig.colorbar(c0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(c2, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        # Original behavior: one panel per output
        for i, axis in enumerate(axes):
            actual_output_idx = output_dim if (ax is not None and output_dim is not None) else i
            vmin = float(ctrl_lower[:, :, i].min().item())
            vmax = float(ctrl_upper[:, :, i].max().item())
            contour = axis.contourf(V1.numpy(), V2.numpy(), 
                                  output_grid[:, :, i].detach().numpy(), 
                                  levels=20, cmap='RdBu', vmin=vmin, vmax=vmax)
            plt.colorbar(contour, ax=axis)
            axis.set_xlabel(dim1_label)
            axis.set_ylabel(dim2_label)
            if ax is None:
                axis.set_title(f'Output {actual_output_idx} at t={time_val:.1f}s')
            axis.grid(True, alpha=0.3)

    fixed_str = ', '.join([f"{system.state_labels[d] if d < len(system.state_labels) else f'dim_{d}'}={float(v):.2f}"
                           for d, v in fixed_dims.items()])
    title = f"{system.__class__.__name__} - NNInput ({role})\n"
    title += f"Resolution: {resolution1}×{resolution2}"
    if fixed_str:
        title += f" | Fixed: {fixed_str}"
    if ax is None:
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()

    return fig


def run_interactive(nn_input, system, vis_cfg, args, tag=None, ref_grid=None):
    if vis_cfg and 'slices' in vis_cfg and len(vis_cfg['slices']) > 0:
        first_slice = vis_cfg['slices'][0]
        dims = first_slice.get('dims', [0, 1])
        initial_fixed = first_slice.get('fixed', {})
        initial_time = first_slice.get('times', [0.0])[0]
    else:
        dims = [0, 1]
        initial_fixed = {}
        initial_time = 0.0

    dim1, dim2 = dims[0], dims[1]
    vis_resolution = vis_cfg.get('resolution', None)

    sliders = []

    if not getattr(nn_input, 'time_invariant', True):
        sliders.append(create_time_slider(np.linspace(0.0, float(getattr(system, 'time_horizon')), 51), description='Time (s)'))
        time_points = True
    else:
        time_points = False

    slider_dims = []
    for dim in range(system.state_dim):
        if dim not in dims:
            min_val = float(system.state_limits[0, dim].item())
            max_val = float(system.state_limits[1, dim].item())
            initial_val = initial_fixed.get(dim, (min_val + max_val) / 2)
            label = system.state_labels[dim] if dim < len(system.state_labels) else f"dim_{dim}"
            sliders.append(SliderSpec(
                name=f'dim_{dim}',
                min_val=min_val,
                max_val=max_val,
                initial_val=initial_val,
                description=label
            ))
            slider_dims.append(dim)

    num_outputs = getattr(nn_input, 'dim', None) or 1
    if num_outputs > 1:
        sliders.append(SliderSpec(
            name='output_dim',
            min_val=0,
            max_val=num_outputs - 1,
            initial_val=0,
            step=1,
            description='Output Dimension'
        ))

    def update_visualization(*slider_values, ax=None):
        if time_points:
            time_val = slider_values[0]
            if num_outputs > 1:
                fixed_vals = slider_values[1:-1]
                output_dim = int(slider_values[-1])
            else:
                fixed_vals = slider_values[1:]
                output_dim = None
        else:
            time_val = initial_time
            if num_outputs > 1:
                fixed_vals = slider_values[:-1]
                output_dim = int(slider_values[-1])
            else:
                fixed_vals = slider_values
                output_dim = None

        fixed_dims = {dim: fixed_vals[i] for i, dim in enumerate(slider_dims)}

        visualize_2d_slice(nn_input, system, time_val, dim1, dim2, fixed_dims, vis_resolution, ax=ax, output_dim=output_dim, ref_grid=ref_grid)

    title_input = tag or 'NNInput'
    title = f"{system.__class__.__name__} - {title_input} (Interactive)"
    viz = InteractiveVisualizer(sliders, update_visualization, title=title, direct_plot=True)
    viz.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', type=str, required=True, help='Tag of the cached NNInput to visualize')
    parser.add_argument('--preset', type=str, default='default', help='Visualization preset name (default: default)')
    parser.add_argument('--save-dir', type=str, help='Optional directory to save figures (default output path used if omitted)')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive visualization with sliders')
    args = parser.parse_args()

    if not args.interactive:
        matplotlib.use('Agg')

    meta = get_nn_input_metadata(args.tag)
    system_name = meta.get('system_name')
    try:
        system = instantiate_system_by_name(system_name)
    except Exception as e:
        print(f"Error: Cannot instantiate system '{system_name}': {e}")
        return
    nn_input = load_nn_input_by_tag(args.tag, system)
    # Try to load the original training GridInput tag from metadata for comparison
    ref_grid = None
    try:
        base = Path('.cache') / 'nn_inputs' / args.tag
        import json as _json
        with open(base.with_suffix('.meta.json'), 'r') as _f:
            raw_meta = _json.load(_f)
        # If checkpoint info is present, print it for user awareness
        ck = raw_meta.get('checkpoint')
        if ck is not None:
            in_prog = bool(raw_meta.get('training_in_progress', False))
            try:
                best_str = f"{float(ck.get('best_mse')):.6f}"
            except Exception:
                best_str = str(ck.get('best_mse'))
            status = 'IN-PROGRESS' if in_prog else 'FINAL'
            print(f"[NNInput] Using cache tag '{args.tag}' ({status}) from checkpoint: epoch={ck.get('epoch')}, best_mse={best_str}")
        if raw_meta.get('input_class') == 'GridInput' and raw_meta.get('input_tag'):
            ref_grid = load_grid_input_by_tag(raw_meta['input_tag'], system, interpolate=True)
    except Exception:
        ref_grid = None

    input_name = meta.get('input_name') or meta.get('input_class') or 'NNInput'
    vis_cfg = load_visualization_presets(system_name, input_name, args.preset)
    if not vis_cfg:
        print(f"Warning: No preset '{args.preset}' found for {system_name}/{input_name}")

    if args.interactive:
        run_interactive(nn_input, system, vis_cfg, args, tag=args.tag, ref_grid=ref_grid)
        return

    figs = []
    filenames = []

    if vis_cfg and 'slices' in vis_cfg:
        print(f"\nGenerating visualizations: {system_name}/{input_name} preset='{args.preset}'")
        vis_resolution = vis_cfg.get('resolution', None)
        for idx, sl in enumerate(vis_cfg['slices']):
            dims = sl.get('dims', [0, 1])
            fixed = sl.get('fixed', {})
            times = sl.get('times', [0.0])
            title_prefix = sl.get('title', '')

            if getattr(nn_input, 'time_invariant', True):
                times = [0.0]

            for tval in times:
                print(f"  Slice {idx+1}: dims={dims}, fixed={fixed}, t={tval}")
                fig = visualize_2d_slice(nn_input, system, tval, dim1=dims[0], dim2=dims[1], fixed_dims=fixed, vis_resolution=vis_resolution, ref_grid=ref_grid)
                if title_prefix:
                    current_title = fig._suptitle.get_text() if fig._suptitle else ''
                    fig.suptitle(f"{title_prefix}\n{current_title}", fontsize=12)
                figs.append(fig)
                fixed_str = '_'.join([f"{k}{v}" for k, v in sorted(fixed.items())])
                fname = f"{system_name}_{input_name}_{args.preset}_slice{idx}_dims{''.join(map(str, dims))}"
                if fixed_str:
                    fname += f"_fix{fixed_str}"
                fname += f"_t{tval:.1f}.png"
                filenames.append(fname)
    else:
        print("\nNo preset found; generating default slice dims=[0,1], t=0.0")
        fig = visualize_2d_slice(nn_input, system, 0.0, dim1=0, dim2=1, ref_grid=ref_grid)
        figs = [fig]
        filenames = [f"{system_name}_{input_name}_default.png"]

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path('outputs') / 'visualizations' / 'nn_inputs' / args.tag / (args.preset or 'default')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(figs)} figure(s) to {save_dir}...")
    for fig, fname in zip(figs, filenames):
        fig.savefig(save_dir / fname, dpi=150, bbox_inches='tight')
        print(f"  ✓ {fname}")
    print(f"\n✓ Visualization complete")


if __name__ == '__main__':
    main()
