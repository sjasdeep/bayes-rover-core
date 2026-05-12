#!/usr/bin/env python3
"""
Visualize GridInput from a tagged cache.

Usage:
  python scripts/grid_input/visualize_grid_input.py --tag TAG [--preset PRESET] [--save-dir DIR]
  python scripts/grid_input/visualize_grid_input.py --tag TAG --interactive  # Interactive mode
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
    get_grid_input_metadata,
    instantiate_system_by_name,
    load_grid_input_by_tag,
)
from src.utils.grids import exact_axis_indices, nearest_time_index, snap_fixed_dims_to_axes
from src.utils.interactive_viz import InteractiveVisualizer, SliderSpec, create_time_slider


def visualize_2d_slice(grid_input, system, time_val, dim1=0, dim2=1, fixed_dims=None, vis_resolution=None, ax=None, fig=None, output_dim=None):
    """
    Visualize a 2D slice of the cached grid.
    
    Args:
        grid_input: GridInput instance
        system: System instance
        time_val: Time value for the slice
        dim1, dim2: State dimensions to visualize
        fixed_dims: Dict of {dim_idx: value} for fixed dimensions
        vis_resolution: List of resolutions for visualization (uses config if None)
        ax: Optional matplotlib axis to plot into (if None, creates new figure)
        fig: Optional matplotlib figure (if None and ax is None, creates new figure)
    
    Returns:
        fig: The matplotlib figure (either provided or newly created)
    """
    if system.state_dim < 2:
        print("System has less than 2 dimensions, cannot create 2D visualization")
        return
    
    # Set defaults for fixed dimensions
    if fixed_dims is None:
        fixed_dims = {}
    for i in range(system.state_dim):
        if i not in [dim1, dim2] and i not in fixed_dims:
            # Use middle of range from state_limits
            lower = system.state_limits[0, i].item()
            upper = system.state_limits[1, i].item()
            if torch.isinf(torch.tensor(lower)) or torch.isinf(torch.tensor(upper)):
                raise ValueError(f"Cannot visualize: dimension {i} has infinite limits")
            fixed_dims[i] = (lower + upper) / 2
    
    # Choose axes sampling based on interpolation flag
    if grid_input.interpolate and vis_resolution is not None:
        # Interpolated visualization with custom resolution
        lower1 = system.state_limits[0, dim1].item()
        upper1 = system.state_limits[1, dim1].item()
        lower2 = system.state_limits[0, dim2].item()
        upper2 = system.state_limits[1, dim2].item()
        resolution1 = int(vis_resolution[dim1])
        resolution2 = int(vis_resolution[dim2])
        vals1 = torch.linspace(lower1, upper1, resolution1)
        vals2 = torch.linspace(lower2, upper2, resolution2)
    else:
        # Exact lookup: use cached grid points for perfect alignment
        vals1 = grid_input._state_grid_points[dim1]
        vals2 = grid_input._state_grid_points[dim2]
        resolution1 = len(vals1)
        resolution2 = len(vals2)
    
    V1, V2 = torch.meshgrid(vals1, vals2, indexing='ij')
    
    # Create state tensor
    states = torch.zeros(resolution1, resolution2, system.state_dim)
    states[:, :, dim1] = V1
    states[:, :, dim2] = V2
    for dim, val in fixed_dims.items():
        states[:, :, dim] = val
    
    # Flatten states for evaluating control limits consistently
    flat_states = states.reshape(-1, system.state_dim)

    if grid_input.interpolate:
        # Evaluate with interpolation over the requested grid
        outputs = grid_input.input(flat_states, time_val)
        output_grid = outputs.reshape(resolution1, resolution2, -1)
        snapped_fixed = {k: float(v) for k, v in (fixed_dims or {}).items()}
        print(f"✓ Evaluated {resolution1}×{resolution2} grid via interpolation")
    else:
        # Slice directly from cache using nearest grid indices for fixed dimensions (snap if needed)
        snapped_fixed = snap_fixed_dims_to_axes({k: v for k, v in fixed_dims.items() if k not in (dim1, dim2)}, grid_input._state_grid_points)
        fixed_indices = {}
        for dim, val in snapped_fixed.items():
            idx = int(exact_axis_indices(grid_input._state_grid_points[dim], torch.tensor([val], dtype=grid_input._state_grid_points[dim].dtype))[0].item())
            fixed_indices[dim] = idx

        cache_indices = []
        for d in range(system.state_dim):
            if d == dim1 or d == dim2:
                cache_indices.append(slice(None))
            else:
                cache_indices.append(fixed_indices.get(d, 0))

        # Handle optional time dimension (snap to nearest)
        if getattr(grid_input, '_time_grid_points', None) is not None:
            t_idx = int(nearest_time_index(grid_input._time_grid_points, float(time_val))[0].item())
            cache_indices.append(t_idx)

        # Input dimension axis
        cache_indices.append(slice(None))

        output_grid = grid_input._grid_cache[tuple(cache_indices)]
        # Ensure ordering is (dim1, dim2, input_dim)
        if dim1 > dim2:
            output_grid = output_grid.transpose(0, 1)
        print(f"✓ Extracted {resolution1}×{resolution2} grid from cache (snapped fixed dims where needed)")
    
    # Reshape for plotting using actual grid resolution
    output_grid = output_grid.reshape(resolution1, resolution2, -1)
    
    # Compute control limits across the grid to enforce consistent colormap scales
    ctrl_lower, ctrl_upper = system.control_limits(flat_states, time_val)
    ctrl_lower = ctrl_lower.reshape(resolution1, resolution2, -1)
    ctrl_upper = ctrl_upper.reshape(resolution1, resolution2, -1)

    # Create or use provided figure/axes
    num_outputs = output_grid.shape[-1]
    
    if ax is None:
        # Static mode: create new figure and axes for all outputs
        if fig is None:
            fig, axes = plt.subplots(1, num_outputs, figsize=(6*num_outputs, 5))
        else:
            axes = fig.subplots(1, num_outputs)
        if num_outputs == 1:
            axes = [axes]
    else:
        # Interactive mode: use provided axis for single output
        if output_dim is not None:
            # Show only the selected output dimension
            axes = [ax]
            # Filter to show only the selected output
            output_grid = output_grid[:, :, output_dim:output_dim+1]
            ctrl_lower = ctrl_lower[:, :, output_dim:output_dim+1]
            ctrl_upper = ctrl_upper[:, :, output_dim:output_dim+1]
        else:
            # No output_dim specified, show first output only
            axes = [ax]
            output_grid = output_grid[:, :, 0:1]
            ctrl_lower = ctrl_lower[:, :, 0:1]
            ctrl_upper = ctrl_upper[:, :, 0:1]
        if fig is None:
            fig = ax.figure
    
    dim1_label = system.state_labels[dim1] if dim1 < len(system.state_labels) else f"dim_{dim1}"
    dim2_label = system.state_labels[dim2] if dim2 < len(system.state_labels) else f"dim_{dim2}"
    
    for i, axis in enumerate(axes):
        # In interactive mode with output_dim slider, i will always be 0 but we want to show the actual output_dim
        actual_output_idx = output_dim if (ax is not None and output_dim is not None) else i
        
        vmin = float(ctrl_lower[:, :, i].min().item())
        vmax = float(ctrl_upper[:, :, i].max().item())
        contour = axis.contourf(V1.numpy(), V2.numpy(), 
                              output_grid[:, :, i].numpy(), 
                              levels=20, cmap='RdBu', vmin=vmin, vmax=vmax)
        
        # For interactive mode with provided axis, we need to handle colorbar carefully
        # Check if this axis already has a colorbar and reuse its axes
        existing_cbar_ax = None
        if ax is not None and fig is not None:
            for cbar_ax in fig.get_axes():
                if cbar_ax.get_label() == '<colorbar>' and cbar_ax != axis:
                    # Check if this colorbar belongs to our axis
                    # by checking if it's adjacent
                    existing_cbar_ax = cbar_ax
                    break
        
        if existing_cbar_ax is not None:
            # Reuse existing colorbar axis
            existing_cbar_ax.clear()
            cbar = plt.colorbar(contour, cax=existing_cbar_ax)
        else:
            # Create new colorbar
            cbar = plt.colorbar(contour, ax=axis)
        
        axis.set_xlabel(dim1_label)
        axis.set_ylabel(dim2_label)
        # Only set axis title in static mode (when ax is None)
        # In interactive mode, the InteractiveVisualizer handles the overall title
        if ax is None:
            axis.set_title(f'Output {actual_output_idx} at t={time_val:.1f}s')
        axis.grid(True, alpha=0.3)
    
    # Add overall title with grid resolution info
    # Display snapped fixed values for clarity
    display_fixed = {}
    for d, v in fixed_dims.items():
        if d in snapped_fixed:
            display_fixed[d] = snapped_fixed[d]
        else:
            display_fixed[d] = v
    fixed_str = ', '.join([f"{system.state_labels[d] if d < len(system.state_labels) else f'dim_{d}'}={display_fixed[d]:.2f}"
                           for d in display_fixed])
    title = f'{system.__class__.__name__} - {grid_input.wrapped_input.__class__.__name__} ({grid_input.type})\n'
    title += f'Grid Resolution: {resolution1}×{resolution2}'
    if fixed_str:
        title += f' | Fixed: {fixed_str}'
    
    # Only set suptitle if we're not using a provided axis (interactive mode handles title separately)
    if ax is None:
        fig.suptitle(title, fontsize=12)
    
    if ax is None:
        plt.tight_layout()
    
    return fig


def visualize_time_evolution(grid_input, system, state, time_points):
    """
    Visualize how control evolves over time for a fixed state.
    
    Args:
        grid_input: GridInput instance
        system: System instance
        state: State tensor [state_dim]
        time_points: Array of time values
    """
    outputs = []
    for t in time_points:
        output = grid_input.input(state.unsqueeze(0), t)
        outputs.append(output.squeeze().numpy())
    
    outputs = np.array(outputs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if outputs.shape[-1] == 1:
        ax.plot(time_points, outputs, 'b-', linewidth=2)
        ax.set_ylabel('Control Output')
    else:
        for i in range(outputs.shape[-1]):
            ax.plot(time_points, outputs[:, i], label=f'Output {i}', linewidth=2)
        ax.legend()
        ax.set_ylabel('Control Outputs')
    
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{system.__class__.__name__} - {grid_input.wrapped_input.__class__.__name__}\n'
                f'State: {state.numpy()}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_interactive(grid_input, system, vis_cfg, args, input_name=None, tag=None):
    """
    Run interactive visualization mode with sliders.
    
    Creates an interactive window where users can adjust:
    - Time (if grid is time-variant)
    - Fixed dimensions (all dimensions not being plotted)
    
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
    if grid_input._time_grid_points is not None:
        time_points = grid_input._time_grid_points.cpu().numpy()
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
    
    # Output dimension slider (discrete selector) if multiple outputs
    # Get the output dimensionality from the cached grid
    num_outputs = grid_input._grid_cache.shape[-1]
    if num_outputs > 1:
        sliders.append(SliderSpec(
            name='output_dim',
            min_val=0,
            max_val=num_outputs - 1,
            initial_val=0,
            step=1,
            description='Output Dimension'
        ))
    
    # Update function
    def update_visualization(*slider_values, ax=None):
        # Parse slider values
        if time_points is not None:
            time_val = slider_values[0]
            if num_outputs > 1:
                fixed_vals = slider_values[1:-1]  # All except first (time) and last (output_dim)
                output_dim = int(slider_values[-1])
            else:
                fixed_vals = slider_values[1:]
                output_dim = None
        else:
            time_val = initial_time
            if num_outputs > 1:
                fixed_vals = slider_values[:-1]  # All except last (output_dim)
                output_dim = int(slider_values[-1])
            else:
                fixed_vals = slider_values
                output_dim = None
        
        # Build fixed_dims dict
        fixed_dims = {}
        for i, dim in enumerate(slider_dims):
            fixed_dims[dim] = fixed_vals[i]
        
        # Call visualization with provided axis
        visualize_2d_slice(grid_input, system, time_val, dim1, dim2, fixed_dims, vis_resolution, ax=ax, output_dim=output_dim)
    
    # Create title - prefer tag (unique identifier), fallback to input_name or class name
    if tag:
        title_input = tag
    elif input_name:
        title_input = input_name
    elif grid_input.wrapped_input is not None:
        title_input = grid_input.wrapped_input.__class__.__name__
    else:
        title_input = "GridInput"
    
    title = f"{system.__class__.__name__} - {title_input} (Interactive)"
    viz = InteractiveVisualizer(sliders, update_visualization, title=title, direct_plot=True)
    
    print(f"\n{'='*60}")
    print("Interactive Mode")
    print('='*60)
    print(f"Plotting dimensions: {dim1} ({system.state_labels[dim1] if dim1 < len(system.state_labels) else f'dim_{dim1}'}), "
          f"{dim2} ({system.state_labels[dim2] if dim2 < len(system.state_labels) else f'dim_{dim2}'})")
    if time_points is not None:
        print(f"Time range: [{time_points[0]:.2f}, {time_points[-1]:.2f}] s")
    print(f"Fixed dimensions: {len(slider_dims)}")
    if num_outputs > 1:
        print(f"Output dimensions: {num_outputs}")
    print("\nAdjust sliders to explore the grid input interactively.")
    print("Close the window when done.\n")
    
    viz.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', type=str, required=True, help='Tag of the cached grid input to visualize')
    parser.add_argument('--preset', type=str, default='default', help='Visualization preset name (default: default)')
    parser.add_argument('--save-dir', type=str, help='Optional directory to save figures (default output path used if omitted)')
    parser.add_argument('--interpolate', action='store_true', help='Enable multilinear interpolation for off-grid visualization')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive visualization with sliders')
    args = parser.parse_args()
    
    # Set backend based on mode
    if not args.interactive:
        matplotlib.use('Agg')  # headless backend for static mode

    # Read metadata and reconstruct system, then load GridInput via cache loader
    meta = get_grid_input_metadata(args.tag)
    system_name = meta['system_name']
    input_name = meta['input_name']
    try:
        system = instantiate_system_by_name(system_name)
    except Exception as e:
        print(f"Error: Cannot instantiate system '{system_name}': {e}")
        return
    grid_input = load_grid_input_by_tag(args.tag, system, interpolate=args.interpolate)
    if args.interpolate:
        print(f"Interpolation enabled: Will use custom resolution from preset if available")
    else:
        print(f"Interpolation disabled: Will use cached grid points directly")

    # Load visualization config using shared helper
    vis_cfg = load_visualization_presets(system_name, input_name, args.preset)
    if not vis_cfg:
        print(f"Warning: No preset '{args.preset}' found for {system_name}/{input_name}")
    
    # Interactive mode
    if args.interactive:
        run_interactive(grid_input, system, vis_cfg, args, input_name=input_name, tag=args.tag)
        return

    figs = []
    filenames = []

    if vis_cfg and 'slices' in vis_cfg:
        print(f"\nGenerating visualizations: {system_name}/{input_name} preset='{args.preset}'")
        # Use resolution only if interpolation is enabled; otherwise, use cached grid points
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

            # If time-invariant, ignore times beyond a single point
            if grid_input._time_grid_points is None:
                times = [0.0]

            for tval in times:
                print(f"  Slice {idx+1}: dims={dims}, fixed={fixed}, t={tval}")
                fig = visualize_2d_slice(grid_input, system, tval, dim1=dims[0], dim2=dims[1], fixed_dims=fixed, vis_resolution=vis_resolution)
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
        fig = visualize_2d_slice(grid_input, system, 0.0, dim1=0, dim2=1)
        figs = [fig]
        filenames = [f"{system_name}_{input_name}_default.png"]

    # Determine save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path('outputs') / 'visualizations' / 'grid_inputs' / args.tag / (args.preset or 'default')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(figs)} figure(s) to {save_dir}...")
    for fig, fname in zip(figs, filenames):
        fig.savefig(save_dir / fname, dpi=150, bbox_inches='tight')
        print(f"  ✓ {fname}")
    print(f"\n✓ Visualization complete")


if __name__ == '__main__':
    main()
