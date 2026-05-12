#!/usr/bin/env python3
"""
Visualize a tagged GridSet cache in 2D slices using presets.

Usage:
  python scripts/grid_set/visualize_grid_set.py --tag TAG [--preset PRESET] [--save-dir DIR] [--interpolate]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impl.inputs.derived.grid_input import GridInput
from src.impl.sets.grid_set import GridSet
from src.utils.cache_loaders import (
    get_grid_set_metadata,
    instantiate_system_by_name,
    load_grid_input_by_tag,
    load_nn_input_by_tag,
    load_grid_set_by_tag,
)
from src.utils.config import load_visualization_presets
from src.utils.grids import nearest_time_index, snap_fixed_dims_to_axes
from src.utils.interactive_viz import InteractiveVisualizer, SliderSpec, create_time_slider


# Nominal sampling now uses GridInput reconstruction when a GridInput payload is available.


def visualize_bounds_2d(
    grid_set: GridSet,
    system,
    time_val,
    dim1=0,
    dim2=1,
    input_dim=0,
    fixed_dims=None,
    vis_resolution=None,
    nominal_grid_input: GridInput | None = None,
    ax=None,
    fig=None,
    equal_aspect: bool = False,
):
    if system.state_dim < 2:
        print("System has less than 2 dimensions, cannot create 2D visualization")
        return

    if fixed_dims is None:
        fixed_dims = {}
    for dim in range(system.state_dim):
        if dim not in (dim1, dim2) and dim not in fixed_dims:
            lower = system.state_limits[0, dim].item()
            upper = system.state_limits[1, dim].item()
            fixed_dims[dim] = (lower + upper) / 2.0

    if grid_set._time_grid_points is None:
        actual_time = 0.0
    else:
        tidx = int(nearest_time_index(grid_set._time_grid_points, float(time_val))[0].item())
        actual_time = float(grid_set._time_grid_points[tidx].item())

    lower1 = system.state_limits[0, dim1].item()
    upper1 = system.state_limits[1, dim1].item()
    lower2 = system.state_limits[0, dim2].item()
    upper2 = system.state_limits[1, dim2].item()

    if not grid_set.interpolate:
        vals1 = grid_set._state_grid_points[dim1]
        vals2 = grid_set._state_grid_points[dim2]
        resolution1 = len(vals1)
        resolution2 = len(vals2)
        print(f"  Using exact grid points: {resolution1}×{resolution2}")
    else:
        if vis_resolution is not None and len(vis_resolution) > max(dim1, dim2):
            resolution1 = int(vis_resolution[dim1])
            resolution2 = int(vis_resolution[dim2])
        else:
            resolution1 = len(grid_set._state_grid_points[dim1])
            resolution2 = len(grid_set._state_grid_points[dim2])
        vals1 = torch.linspace(lower1, upper1, resolution1, device=grid_set.device)
        vals2 = torch.linspace(lower2, upper2, resolution2, device=grid_set.device)
        print(f"  Using interpolated sampling: {resolution1}×{resolution2}")

    V1, V2 = torch.meshgrid(vals1, vals2, indexing='ij')
    vis_states = torch.zeros(resolution1, resolution2, system.state_dim, device=grid_set.device)
    vis_states[:, :, dim1] = V1
    vis_states[:, :, dim2] = V2
    for dim, val in fixed_dims.items():
        vis_states[:, :, dim] = val

    if not grid_set.interpolate:
        snapped = snap_fixed_dims_to_axes({k: v for k, v in fixed_dims.items() if k not in (dim1, dim2)}, grid_set._state_grid_points)
        for dim, snapped_val in snapped.items():
            orig = fixed_dims.get(dim, snapped_val)
            if abs(snapped_val - orig) > 1e-10:
                print(f"  Snapped fixed dim {dim} from {orig:.6f} to {snapped_val:.6f}")
            vis_states[:, :, dim] = snapped_val
            fixed_dims[dim] = snapped_val

    flat_vis_states = vis_states.reshape(-1, system.state_dim)
    print(f"  Evaluating {len(flat_vis_states)} visualization points...")
    lower_bounds, upper_bounds = grid_set.as_box(flat_vis_states, actual_time)
    lower_col = lower_bounds[:, input_dim].detach().cpu()
    upper_col = upper_bounds[:, input_dim].detach().cpu()

    lower_grid = lower_col.reshape(resolution1, resolution2)
    upper_grid = upper_col.reshape(resolution1, resolution2)
    width_grid = (upper_col - lower_col).reshape(resolution1, resolution2)

    print(f"  Bounds statistics for input dim {input_dim}:")
    print(f"    Lower:  min={float(lower_grid.min().item()):.3f}, max={float(lower_grid.max().item()):.3f}, mean={float(lower_grid.mean().item()):.3f}")
    print(f"    Upper:  min={float(upper_grid.min().item()):.3f}, max={float(upper_grid.max().item()):.3f}, mean={float(upper_grid.mean().item()):.3f}")
    print(f"    Width:  min={float(width_grid.min().item()):.3f}, max={float(width_grid.max().item()):.3f}, mean={float(width_grid.mean().item()):.3f}")

    # Setup figure and axes
    if ax is None:
        # Static mode: create new figure with 4 subplots
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    else:
        # Interactive mode: use provided figure
        if fig is None:
            fig = ax.figure
        
        # Check if we already have 4 subplot axes (excluding slider axes and colorbars)
        plot_axes = [a for a in fig.get_axes() if a.get_label() not in ['<colorbar_0>', '<colorbar_1>', '<colorbar_2>', '<colorbar_3>'] 
                     and not hasattr(a, '_is_slider') and a != ax]
        
        if len(plot_axes) == 4:
            # Reuse existing axes, just clear them
            axes = plot_axes
            for axis in axes:
                axis.clear()
        else:
            # First time: need to create 4 subplot axes within the provided plot area
            # Hide the placeholder axis but don't remove it
            ax.set_visible(False)
            
            # Get the position of the provided axis
            pos = ax.get_position()
            
            # Create 4 subplots in the same position
            subplot_width = pos.width / 4
            axes = []
            for i in range(4):
                subplot_ax = fig.add_axes([pos.x0 + i * subplot_width, pos.y0, subplot_width * 0.95, pos.height])
                axes.append(subplot_ax)
    
    dim1_label = system.state_labels[dim1] if dim1 < len(system.state_labels) else f"dim_{dim1}"
    dim2_label = system.state_labels[dim2] if dim2 < len(system.state_labels) else f"dim_{dim2}"

    # Evaluate system control limits over the same grid to fix consistent colormap limits
    ctrl_lower, ctrl_upper = system.control_limits(flat_vis_states, actual_time)
    ctrl_lower = ctrl_lower[:, input_dim].reshape(resolution1, resolution2)
    ctrl_upper = ctrl_upper[:, input_dim].reshape(resolution1, resolution2)
    vmin_ctrl = float(ctrl_lower.min().item())
    vmax_ctrl = float(ctrl_upper.max().item())
    vmax_width = float((ctrl_upper - ctrl_lower).max().item())

    # Use contourf for smooth rendering (consistent with GridInput visualizer)
    c0 = axes[0].contourf(V1.cpu().numpy(), V2.cpu().numpy(), lower_grid.cpu().numpy(), levels=20, cmap='RdBu', vmin=vmin_ctrl, vmax=vmax_ctrl)
    axes[0].set_xlabel(dim1_label)
    axes[0].set_ylabel(dim2_label)
    # Simplified titles in interactive mode, detailed in static mode
    if ax is None:
        axes[0].set_title(f'Lower Bound (Input dim {input_dim})')
    else:
        axes[0].set_title('Lower Bound')
    
    # Find existing colorbar axes for reuse in interactive mode
    existing_cbar_axes = {}
    if ax is not None:
        for i, cbar_ax in enumerate(fig.get_axes()):
            if cbar_ax.get_label() == f'<colorbar_{i}>':
                existing_cbar_axes[i] = cbar_ax
    
    if 0 in existing_cbar_axes:
        existing_cbar_axes[0].clear()
        plt.colorbar(c0, cax=existing_cbar_axes[0])
        existing_cbar_axes[0].set_label('<colorbar_0>')
    else:
        cbar0 = plt.colorbar(c0, ax=axes[0])
        if ax is not None:
            cbar0.ax.set_label('<colorbar_0>')

    c1 = axes[1].contourf(V1.cpu().numpy(), V2.cpu().numpy(), upper_grid.cpu().numpy(), levels=20, cmap='RdBu', vmin=vmin_ctrl, vmax=vmax_ctrl)
    axes[1].set_xlabel(dim1_label)
    axes[1].set_ylabel(dim2_label)
    if ax is None:
        axes[1].set_title(f'Upper Bound (Input dim {input_dim})')
    else:
        axes[1].set_title('Upper Bound')
    
    if 1 in existing_cbar_axes:
        existing_cbar_axes[1].clear()
        plt.colorbar(c1, cax=existing_cbar_axes[1])
        existing_cbar_axes[1].set_label('<colorbar_1>')
    else:
        cbar1 = plt.colorbar(c1, ax=axes[1])
        if ax is not None:
            cbar1.ax.set_label('<colorbar_1>')

    c2 = axes[2].contourf(V1.cpu().numpy(), V2.cpu().numpy(), width_grid.cpu().numpy(), levels=20, cmap='inferno', vmin=0.0, vmax=vmax_width)
    axes[2].set_xlabel(dim1_label)
    axes[2].set_ylabel(dim2_label)
    if ax is None:
        axes[2].set_title(f'Uncertainty Width (Input dim {input_dim})')
    else:
        axes[2].set_title('Uncertainty Width')
    
    if 2 in existing_cbar_axes:
        existing_cbar_axes[2].clear()
        plt.colorbar(c2, cax=existing_cbar_axes[2])
        existing_cbar_axes[2].set_label('<colorbar_2>')
    else:
        cbar2 = plt.colorbar(c2, ax=axes[2])
        if ax is not None:
            cbar2.ax.set_label('<colorbar_2>')

    # Nominal control: use provided GridInput (if any) and its input() API
    if nominal_grid_input is not None:
        outputs = nominal_grid_input.input(flat_vis_states.cpu(), time_val)
        nominal_vals = outputs.detach().cpu()
        nominal_grid = nominal_vals[:, input_dim].reshape(resolution1, resolution2).numpy()
        c3 = axes[3].contourf(V1.cpu().numpy(), V2.cpu().numpy(), nominal_grid, levels=20, cmap='RdBu', vmin=vmin_ctrl, vmax=vmax_ctrl)
        if ax is None:
            axes[3].set_title(f'Nominal Input (Input dim {input_dim})')
        else:
            axes[3].set_title('Nominal Input')
    else:
        midpoint_grid = ((lower_col + upper_col) * 0.5).reshape(resolution1, resolution2).numpy()
        c3 = axes[3].contourf(V1.cpu().numpy(), V2.cpu().numpy(), midpoint_grid, levels=20, cmap='RdBu', vmin=vmin_ctrl, vmax=vmax_ctrl)
        if ax is None:
            axes[3].set_title(f'Box Midpoint (Input dim {input_dim})')
        else:
            axes[3].set_title('Box Midpoint')
    axes[3].set_xlabel(dim1_label)
    axes[3].set_ylabel(dim2_label)
    
    if 3 in existing_cbar_axes:
        existing_cbar_axes[3].clear()
        plt.colorbar(c3, cax=existing_cbar_axes[3])
        existing_cbar_axes[3].set_label('<colorbar_3>')
    else:
        cbar3 = plt.colorbar(c3, ax=axes[3])
        if ax is not None:
            cbar3.ax.set_label('<colorbar_3>')

    fixed_str = ', '.join([f"{system.state_labels[d] if d < len(system.state_labels) else f'dim_{d}'}={v:.2f}" for d, v in sorted(fixed_dims.items()) if d not in (dim1, dim2)])
    title = f"{system.__class__.__name__} - GridSet Bounds at t={actual_time:.2f} ({grid_set.set_type} type)"
    if fixed_str:
        title += f" | Fixed: {fixed_str}"
    
    # Apply equal aspect ratio if requested
    if equal_aspect:
        try:
            for a in axes:
                a.set_aspect('equal', adjustable='box')
        except Exception:
            pass

    # Only set suptitle if we're not using a provided axis (interactive mode handles title separately)
    if ax is None:
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
    
    return fig


def run_interactive(grid_set, system, vis_cfg, args, nominal_grid_input=None, grid_input_tag=None, nn_input_tag=None, set_type=None):
    """
    Run interactive visualization mode with sliders.
    
    Creates an interactive window where users can adjust:
    - Time (if grid is time-variant)
    - Fixed dimensions (all dimensions not being plotted)
    - Input dimension selector
    
    The plotted dimensions are taken from the first slice in the preset,
    or default to [0, 1] if no preset is provided.
    """
    
    # Determine dimensions to plot and fixed dimensions
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
    vis_resolution = vis_cfg.get('resolution', None) if args.interpolate else None
    
    # Build sliders
    sliders = []
    
    # Time slider if time-variant
    if grid_set._time_grid_points is not None:
        time_points = grid_set._time_grid_points.cpu().numpy()
        sliders.append(create_time_slider(time_points, description='Time (s)'))
    else:
        time_points = None
    
    # Sliders for fixed dimensions
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
    
    # Input dimension slider (discrete selector)
    sliders.append(SliderSpec(
        name='input_dim',
        min_val=0,
        max_val=grid_set.input_dim - 1,
        initial_val=0,
        step=1,
        description='Input Dimension'
    ))
    
    # Update function
    def update_visualization(*slider_values, ax=None):
        # Parse slider values
        if time_points is not None:
            time_val = slider_values[0]
            fixed_vals = slider_values[1:-1]  # All except first (time) and last (input_dim)
            input_dim = int(slider_values[-1])
        else:
            time_val = initial_time
            fixed_vals = slider_values[:-1]  # All except last (input_dim)
            input_dim = int(slider_values[-1])
        
        # Build fixed_dims dict
        fixed_dims = {}
        for i, dim in enumerate(slider_dims):
            fixed_dims[dim] = fixed_vals[i]
        
        # Call visualization with provided axis
        visualize_bounds_2d(
            grid_set,
            system,
            time_val,
            dim1,
            dim2,
            input_dim,
            fixed_dims,
            vis_resolution,
            nominal_grid_input,
            ax=ax,
            equal_aspect=args.equal_aspect,
        )
    
    # Create title with GridInput tag and set type information
    title_parts = [system.__class__.__name__, "GridSet Bounds"]
    src_tag = nn_input_tag or grid_input_tag
    if src_tag:
        title_parts.append(f"(from {src_tag})")
    if set_type:
        title_parts.append(f"[{set_type}]")
    title = " - ".join(title_parts) + " (Interactive)"
    
    viz = InteractiveVisualizer(sliders, update_visualization, title=title, 
                               direct_plot=True, figsize=(18, 6))
    
    print(f"\n{'='*60}")
    print("Interactive Mode")
    print('='*60)
    print(f"Plotting dimensions: {dim1} ({system.state_labels[dim1] if dim1 < len(system.state_labels) else f'dim_{dim1}'}), "
          f"{dim2} ({system.state_labels[dim2] if dim2 < len(system.state_labels) else f'dim_{dim2}'})")
    if time_points is not None:
        print(f"Time range: [{time_points[0]:.2f}, {time_points[-1]:.2f}] s")
    print(f"Fixed dimensions: {len(slider_dims)}")
    print(f"Input dimensions: {grid_set.input_dim}")
    print("\nAdjust sliders to explore the grid set bounds interactively.")
    print("Close the window when done.\n")
    
    viz.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', type=str, required=True, help='GridSet cache tag to visualize')
    parser.add_argument('--preset', type=str, default='default', help='Visualization preset name')
    parser.add_argument('--save-dir', type=str, help='Optional directory to save figures')
    parser.add_argument('--interpolate', action='store_true', help='Interpolate box bounds across grid')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive visualization with sliders')
    parser.add_argument('--equal-aspect', action='store_true', help='Force equal aspect ratio on 2D subplot axes')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for saved figures (and interactive display)')
    args = parser.parse_args()

    # Set matplotlib backend based on mode
    if args.interactive:
        # Interactive mode needs TkAgg or similar
        try:
            matplotlib.use('TkAgg')
        except:
            try:
                matplotlib.use('Qt5Agg')
            except:
                print("Warning: Could not set interactive backend")
        # Apply requested DPI to interactive display
        try:
            matplotlib.rcParams['figure.dpi'] = args.dpi
        except Exception:
            pass
    else:
        # Static mode uses Agg for saving files
        matplotlib.use('Agg')

    meta = get_grid_set_metadata(args.tag)
    system_name = meta['system_name']
    input_name = meta['input_name']
    set_type = meta['set_type']

    system = instantiate_system_by_name(system_name)

    grid_set = load_grid_set_by_tag(args.tag, system, interpolate=args.interpolate)
    if args.interpolate:
        print(f"Interpolation enabled: Will use custom resolution from preset if available")
    else:
        print(f"Interpolation disabled: Will use cached grid points directly")

    # Load GridInput if available (for nominal control)
    nominal_grid_input = None
    grid_input_tag = meta.get('grid_input_tag')
    nn_input_tag = meta.get('nn_input_tag')
    if grid_input_tag:
        nominal_grid_input = load_grid_input_by_tag(grid_input_tag, system, interpolate=args.interpolate)
        print(f"Loaded GridInput for nominal plotting: {grid_input_tag}")
    elif nn_input_tag:
        # Load NNInput as nominal when GridSet was constructed from a neural input
        try:
            nominal_grid_input = load_nn_input_by_tag(nn_input_tag, system)
            print(f"Loaded NNInput for nominal plotting: {nn_input_tag}")
        except Exception as e:
            print(f"Warning: Failed to load NNInput '{nn_input_tag}' for nominal plotting: {e}")

    # Load visualization config using shared helper
    vis_cfg = load_visualization_presets(system_name, input_name, args.preset)
    if not vis_cfg:
        print(f"Warning: No preset '{args.preset}' found for {system_name}/{input_name}")

    # Interactive mode
    if args.interactive:
        run_interactive(grid_set, system, vis_cfg, args, nominal_grid_input, 
                       grid_input_tag=grid_input_tag, nn_input_tag=nn_input_tag, set_type=set_type)
        return

    figs = []
    filenames = []

    if vis_cfg and 'slices' in vis_cfg:
        print(f"\nGenerating visualizations: {system_name}/{input_name} preset='{args.preset}'")
        vis_resolution = vis_cfg.get('resolution', None) if args.interpolate else None
        if vis_resolution is not None:
            print(f"Using custom resolution from preset: {vis_resolution}")
        elif args.interpolate:
            print(f"No custom resolution in preset, will use cached grid resolution for interpolation")
        for idx, sl in enumerate(vis_cfg['slices']):
            dims = sl.get('dims', [0, 1])
            if 'dims' not in sl:
                print(f"⚠ Warning: Slice missing 'dims', using default {dims}")
            fixed = sl.get('fixed', {})
            times = sl.get('times', [0.0])
            if 'times' not in sl:
                print(f"⚠ Warning: Slice missing 'times', using default {times}")
            title_prefix = sl.get('title', '')

            if grid_set._time_grid_points is None:
                times = [0.0]

            for tval in times:
                print(f"  Slice {idx+1}: dims={dims}, fixed={fixed}, t={tval}")
                for input_dim in range(grid_set.input_dim):
                    fig = visualize_bounds_2d(
                        grid_set,
                        system,
                        tval,
                        dim1=dims[0],
                        dim2=dims[1],
                        input_dim=input_dim,
                        fixed_dims=fixed,
                        vis_resolution=vis_resolution,
                        nominal_grid_input=nominal_grid_input,
                        equal_aspect=args.equal_aspect,
                    )
                    if title_prefix:
                        current_title = fig._suptitle.get_text() if fig._suptitle else ''
                        fig.suptitle(f"{title_prefix}\n{current_title}", fontsize=12)
                    figs.append(fig)
                    fixed_str = '_'.join([f"{k}{v}" for k, v in sorted(fixed.items())])
                    fname = f"{system_name}_{input_name}_{args.preset}_slice{idx}_dims{''.join(map(str, dims))}"
                    if fixed_str:
                        fname += f"_fix{fixed_str}"
                    fname += f"_indim{input_dim}_t{tval:.1f}.png"
                    filenames.append(fname)
    else:
        print("\nNo preset found; generating default slice dims=[0,1], t=0.0")
        for input_dim in range(grid_set.input_dim):
            fig = visualize_bounds_2d(
                grid_set,
                system,
                0.0,
                dim1=0,
                dim2=1,
                input_dim=input_dim,
                nominal_grid_input=nominal_grid_input,
                equal_aspect=args.equal_aspect,
            )
            figs.append(fig)
            filenames.append(f"{system_name}_{input_name}_default_indim{input_dim}.png")

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path('outputs') / 'visualizations' / 'grid_sets' / args.tag / (args.preset or 'default')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(figs)} figure(s) to {save_dir}...")
    for fig, fname in zip(figs, filenames):
        fig.savefig(save_dir / fname, dpi=args.dpi, bbox_inches='tight')
        print(f"  ✓ {fname}")
    print(f"\n✓ Visualization complete")


if __name__ == '__main__':
    main()
