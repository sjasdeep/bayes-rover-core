#!/usr/bin/env python3
"""
Visualize GridValue caches by tag.

This script generates visualizations of the HJ reachability grid value (value function)
using 2D slices and optional presets defined in config/visualizations.yaml.

Usage:
  python scripts/grid_value/visualize_grid_value.py \
      --tag {TAG} \
      [--preset {PRESET}] \
      [--save-dir {SAVE_DIR}] \
      [--interpolate]

If --preset is not provided, a simple default visualization will be produced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.systems import System
from src.impl.values.grid_value import GridValue
from src.utils.cache_loaders import (
    get_grid_value_metadata,
    instantiate_system_by_name,
    load_grid_value_by_tag,
)
from src.utils.config import load_visualization_presets
from src.utils.grids import nearest_axis_indices, nearest_time_index
from src.utils.interactive_viz import InteractiveVisualizer, SliderSpec, create_time_slider


def visualize_value_slice_2d(
    vf: GridValue,
    system: System,
    time_val: float,
    slice_dim: int = 2,
    slice_value: Optional[float] = None,
    ax=None,
    fig=None
):
    """
    Visualize a single 2D slice of the grid value at a specific time.
    
    Args:
        vf: GridValue instance
        system: System instance (for obstacles)
        time_val: Time value to visualize
        slice_dim: Dimension to slice (for 3D+ state spaces)
        slice_value: Value at which to slice (None = middle)
        ax: Matplotlib axis to plot into (for interactive mode)
        fig: Matplotlib figure (for interactive mode)
    
    Returns:
        fig: Matplotlib figure
    """
    
    if vf.state_dim < 2:
        print("Cannot create 2D slices for 1D state space")
        return None
    
    # Find nearest time index
    # NOTE: nearest_time_index uses searchsorted which assumes ascending order.
    # For HJ reachability (backward in time), times are typically descending.
    # Use robust argmin approach instead.
    time_diffs = torch.abs(vf._times - float(time_val))
    time_idx = int(torch.argmin(time_diffs).item())
    actual_time = float(vf._times[time_idx].item())
    
    # Determine slice index for 3D state spaces
    if vf.state_dim > 2:
        if slice_value is None:
            slice_idx = vf.grid_shape[slice_dim] // 2
        else:
            coord_t = vf._axes[slice_dim]
            slice_idx = int(nearest_axis_indices(coord_t, torch.tensor([float(slice_value)], dtype=coord_t.dtype, device=coord_t.device))[0].item())
        slice_val = float(vf._axes[slice_dim][slice_idx].item())
    
    # Setup figure and axis
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        else:
            ax = fig.add_subplot(111)
    else:
        if fig is None:
            fig = ax.figure
        ax.clear()
    
    # Get grid value at this time
    value_slice = vf._values[..., time_idx]
    
    # Extract 2D slice
    if vf.state_dim == 2:
        value_2d = value_slice
        X, Y = np.meshgrid(
            vf._axes[0].detach().cpu().numpy(),
            vf._axes[1].detach().cpu().numpy(),
            indexing='ij'
        )
        xlabel, ylabel = 'x', 'y'
    elif vf.state_dim == 3:
        if slice_dim == 2:
            value_2d = value_slice[:, :, slice_idx]
            X, Y = np.meshgrid(
                vf._axes[0].detach().cpu().numpy(),
                vf._axes[1].detach().cpu().numpy(),
                indexing='ij'
            )
            xlabel, ylabel = 'x', 'y'
        elif slice_dim == 1:
            value_2d = value_slice[:, slice_idx, :]
            X, Y = np.meshgrid(
                vf._axes[0].detach().cpu().numpy(),
                vf._axes[2].detach().cpu().numpy(),
                indexing='ij'
            )
            xlabel, ylabel = 'x', 'θ'
        else:  # slice_dim == 0
            value_2d = value_slice[slice_idx, :, :]
            X, Y = np.meshgrid(
                vf._axes[1].detach().cpu().numpy(),
                vf._axes[2].detach().cpu().numpy(),
                indexing='ij'
            )
            xlabel, ylabel = 'y', 'θ'
    else:
        print(f"Visualization not supported for {vf.state_dim}D state spaces")
        return None
    
    # Convert to numpy
    if isinstance(value_2d, torch.Tensor):
        value_2d_np = value_2d.detach().cpu().numpy()
    else:
        value_2d_np = np.asarray(value_2d)
    
    # Determine color normalization
    from matplotlib.colors import TwoSlopeNorm
    vabs = max(abs(float(np.min(value_2d_np))), abs(float(np.max(value_2d_np))))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    
    # Plot filled contours
    levels = np.linspace(-vabs, vabs, 21)
    cf = ax.contourf(X, Y, value_2d_np, levels=levels, cmap='RdYlBu', norm=norm)
    
    # Zero level set (boundary of reachable set)
    ax.contour(X, Y, value_2d_np, levels=[0.0], colors='black', linewidths=2, linestyles='-')
    
    # Add obstacles (if 2D spatial)
    if vf.state_dim >= 2 and slice_dim >= 2:
        from src.utils.obstacles import draw_obstacles_2d
        draw_obstacles_2d(ax, system, zorder=10)
    
    # Handle colorbar
    if ax is not None:
        # Interactive mode: check for existing colorbar
        existing_cbar_ax = None
        for cbar_ax in fig.get_axes():
            if cbar_ax.get_label() == '<colorbar>' and cbar_ax != ax:
                existing_cbar_ax = cbar_ax
                break
        
        if existing_cbar_ax is not None:
            existing_cbar_ax.clear()
            sm = plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu')
            sm.set_array([])
            plt.colorbar(sm, cax=existing_cbar_ax, label='Value')
            existing_cbar_ax.set_label('<colorbar>')
        else:
            sm = plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu')
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Value')
            if fig is not None:
                cbar.ax.set_label('<colorbar>')
    
    # Format
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Only set title in static mode
    if ax is None:
        if vf.state_dim > 2:
            dim_names = ['x', 'y', 'θ']
            slice_name = dim_names[slice_dim] if slice_dim < len(dim_names) else f'dim{slice_dim}'
            ax.set_title(f'Backward Reachable Tube at t={actual_time:.2f}s ({slice_name}={slice_val:.2f})')
        else:
            ax.set_title(f'Backward Reachable Tube at t={actual_time:.2f}s')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if ax is None:
        plt.tight_layout()
    
    return fig


def visualize_2d_slices(
    vf: GridValue,
    system: System,
    time_indices: List[int],
    output_dir: Path,
    slice_dim: int = 2,
    slice_value: Optional[float] = None
):
    """
    Visualize 2D slices of the grid value (value function) at different times.
    
    Args:
    vf: GridValue instance
        system: System instance (for obstacles)
        time_indices: List of time indices to visualize
        output_dir: Output directory for plots
        slice_dim: Dimension to slice (for 3D state spaces)
        slice_value: Value at which to slice (None = middle)
    """
    
    print(f"\nGenerating 2D slice visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if vf.state_dim < 2:
        print("Cannot create 2D slices for 1D state space")
        return
    
    # Determine slice index for 3D state spaces
    if vf.state_dim > 2:
        if slice_value is None:
            slice_idx = vf.grid_shape[slice_dim] // 2
        else:
            coord_t = vf._axes[slice_dim]
            slice_idx = int(nearest_axis_indices(coord_t, torch.tensor([float(slice_value)], dtype=coord_t.dtype, device=coord_t.device))[0].item())
        slice_val = float(vf._axes[slice_dim][slice_idx].item())
        print(f"  Slicing at dimension {slice_dim}, index {slice_idx}, value {slice_val:.3f}")
    
    # Create figure
    n_times = len(time_indices)
    fig, axes = plt.subplots(1, n_times, figsize=(5*n_times, 5))
    
    if n_times == 1:
        axes = [axes]
    
    # Determine symmetric color normalization centered at 0 across all requested time slices
    from matplotlib.colors import TwoSlopeNorm
    vmin_all, vmax_all = None, None
    for time_idx in time_indices:
        value_slice = vf._values[..., time_idx]
        if vf.state_dim == 2:
            value_2d_tmp = value_slice
        elif vf.state_dim == 3:
            if slice_dim == 2:
                value_2d_tmp = value_slice[:, :, slice_idx]
            elif slice_dim == 1:
                value_2d_tmp = value_slice[:, slice_idx, :]
            else:
                value_2d_tmp = value_slice[slice_idx, :, :]
        else:
            continue
        # Ensure numpy arrays for reduction operations
        if isinstance(value_2d_tmp, torch.Tensor):
            value_2d_arr = value_2d_tmp.detach().cpu().numpy()
        else:
            value_2d_arr = np.asarray(value_2d_tmp)
        vmin_cur = float(np.min(value_2d_arr))
        vmax_cur = float(np.max(value_2d_arr))
        vmin_all = vmin_cur if vmin_all is None else min(vmin_all, vmin_cur)
        vmax_all = vmax_cur if vmax_all is None else max(vmax_all, vmax_cur)
    vabs = max(abs(vmin_all if vmin_all is not None else 0.0), abs(vmax_all if vmax_all is not None else 0.0))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        
        # Get grid value at this time (time-last storage)
        value_slice = vf._values[..., time_idx]
        
        # Extract 2D slice if needed
        if vf.state_dim == 2:
            value_2d = value_slice
            X, Y = np.meshgrid(
                vf._axes[0].detach().cpu().numpy(),
                vf._axes[1].detach().cpu().numpy(),
                indexing='ij'
            )
            xlabel, ylabel = 'x', 'y'
        elif vf.state_dim == 3:
            if slice_dim == 2:
                value_2d = value_slice[:, :, slice_idx]
                X, Y = np.meshgrid(
                    vf._axes[0].detach().cpu().numpy(),
                    vf._axes[1].detach().cpu().numpy(),
                    indexing='ij'
                )
                xlabel, ylabel = 'x', 'y'
            elif slice_dim == 1:
                value_2d = value_slice[:, slice_idx, :]
                X, Y = np.meshgrid(
                    vf._axes[0].detach().cpu().numpy(),
                    vf._axes[2].detach().cpu().numpy(),
                    indexing='ij'
                )
                xlabel, ylabel = 'x', 'θ'
            else:  # slice_dim == 0
                value_2d = value_slice[slice_idx, :, :]
                X, Y = np.meshgrid(
                    vf._axes[1].detach().cpu().numpy(),
                    vf._axes[2].detach().cpu().numpy(),
                    indexing='ij'
                )
                xlabel, ylabel = 'y', 'θ'
        else:
            print(f"Visualization not supported for {vf.state_dim}D state spaces")
            return
        
        # Convert to numpy for matplotlib
        if isinstance(value_2d, torch.Tensor):
            value_2d_np = value_2d.detach().cpu().numpy()
        else:
            value_2d_np = np.asarray(value_2d)

        # Plot filled contours with colorbar centered at 0
        levels = np.linspace(-vabs, vabs, 21)
        cf = ax.contourf(X, Y, value_2d_np, levels=levels, cmap='RdYlBu', norm=norm)
        
        # Zero level set (boundary of reachable set)
        ax.contour(X, Y, value_2d_np, levels=[0.0], colors='black', linewidths=2, linestyles='-')
        
        # Add obstacles (if 2D spatial)
        if vf.state_dim >= 2 and slice_dim >= 2:
            from src.utils.obstacles import draw_obstacles_2d
            draw_obstacles_2d(ax, system, zorder=10)
        
        # Format
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f't = {float(vf._times[time_idx].item()):.2f}s')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
    # Single shared colorbar for consistency across subplots
    sm = plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu')
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='Value', shrink=0.9)
    
    # Overall title
    if vf.state_dim > 2:
        dim_names = ['x', 'y', 'θ']
        slice_name = dim_names[slice_dim] if slice_dim < len(dim_names) else f'dim{slice_dim}'
        plt.suptitle(f'Backward Reachable Tube ({slice_name} = {slice_val:.2f})', fontsize=14)
    else:
        plt.suptitle('Backward Reachable Tube', fontsize=14)
    
    output_file = output_dir / 'value_function_slices.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def visualize_reachable_set_evolution(
    vf: GridValue,
    output_dir: Path,
    slice_dim: int = 2,
    slice_value: Optional[float] = None,
    num_frames: int = 10
):
    """
    Visualize evolution of reachable set boundary over time.
    
    Args:
        vf: ValueFunction instance
        output_dir: Output directory
        slice_dim: Dimension to slice (for 3D state spaces)
        slice_value: Value at which to slice (None = middle)
        num_frames: Number of time frames to show
    """
    
    print(f"\nGenerating reachable set evolution visualization...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if vf.state_dim < 2:
        print("Cannot visualize evolution for 1D state space")
        return
    
    # Determine slice index for 3D state spaces
    if vf.state_dim > 2:
        if slice_value is None:
            slice_idx = vf.grid_shape[slice_dim] // 2
        else:
            coord_t = vf._axes[slice_dim]
            slice_idx = int(nearest_axis_indices(coord_t, torch.tensor([float(slice_value)], dtype=coord_t.dtype, device=coord_t.device))[0].item())
    
    # Select time indices
    time_indices = np.linspace(0, int(vf._times.shape[0]) - 1, num_frames, dtype=int)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Get meshgrid
    if vf.state_dim == 2:
        X, Y = np.meshgrid(
            vf._axes[0].detach().cpu().numpy(),
            vf._axes[1].detach().cpu().numpy(),
            indexing='ij'
        )
        xlabel, ylabel = 'x (m)', 'y (m)'
    elif vf.state_dim == 3 and slice_dim == 2:
        X, Y = np.meshgrid(
            vf._axes[0].detach().cpu().numpy(),
            vf._axes[1].detach().cpu().numpy(),
            indexing='ij'
        )
        xlabel, ylabel = 'x (m)', 'y (m)'
    else:
        print("Visualization only supported for 2D or 3D (sliced at θ) spaces")
        return
    
    # Plot zero level sets at different times
    cmap = plt.cm.viridis
    colors = [cmap(i / (num_frames - 1)) for i in range(num_frames)]
    
    for i, time_idx in enumerate(time_indices):
        value_slice = vf._values[..., time_idx]
        
        # Extract 2D slice if needed
        if vf.state_dim == 2:
            value_2d = value_slice
        elif vf.state_dim == 3:
            value_2d = value_slice[:, :, slice_idx]
        
        # Convert to numpy for matplotlib
        if isinstance(value_2d, torch.Tensor):
            value_2d_np = value_2d.detach().cpu().numpy()
        else:
            value_2d_np = np.asarray(value_2d)
        
        # Plot zero level set
        cs = ax.contour(
            X, Y, value_2d_np,
            levels=[0.0],
            colors=[colors[i]],
            linewidths=2,
            linestyles='-'
        )
        
        # Label
        if i % max(1, num_frames // 5) == 0:  # Label every few contours
            ax.clabel(cs, fmt=f't={float(vf._times[time_idx].item()):.2f}s', inline=True, fontsize=8)
    
    # Format
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Reachable Set Evolution')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=Normalize(vmin=float(vf._times[0].item()), vmax=float(vf._times[-1].item()))
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time (s)')
    
    output_file = output_dir / 'reachable_set_evolution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def run_interactive(vf, system, args, tag=None):
    """
    Run interactive visualization mode with sliders.
    
    Creates an interactive window where users can adjust:
    - Time (to scrub through the backward reachable tube evolution)
    - Slice dimension value (for 3D+ state spaces)
    
    Args:
        vf: GridValue instance
        system: System instance
        args: Command line arguments
        tag: GridValue cache tag
    """
    
    # Build sliders
    sliders = []
    
    # Time slider - always available for grid values
    time_points = vf._times.cpu().numpy()
    sliders.append(create_time_slider(time_points, description='Time (s)'))
    
    # Slice value slider for 3D+ state spaces
    slice_dim = args.slice_dim if hasattr(args, 'slice_dim') else 2
    if vf.state_dim > 2:
        slice_axis = vf._axes[slice_dim].cpu().numpy()
        dim_names = ['x', 'y', 'θ']
        slice_label = dim_names[slice_dim] if slice_dim < len(dim_names) else f'dim_{slice_dim}'
        
        initial_slice_val = args.slice_value if hasattr(args, 'slice_value') and args.slice_value is not None else slice_axis[len(slice_axis)//2]
        
        sliders.append(SliderSpec(
            name='slice_value',
            min_val=float(slice_axis.min()),
            max_val=float(slice_axis.max()),
            initial_val=float(initial_slice_val),
            description=f'{slice_label} slice'
        ))
    
    # Update function
    def update_visualization(*slider_values, ax=None):
        time_val = slider_values[0]
        if vf.state_dim > 2:
            slice_value = slider_values[1]
        else:
            slice_value = None
        
        # Call visualization with provided axis
        visualize_value_slice_2d(vf, system, time_val, slice_dim, slice_value, ax=ax)
    
    # Create title
    if tag:
        title = f"{system.__class__.__name__} - {tag} - Reachable Tube (Interactive)"
    else:
        title = f"{system.__class__.__name__} - Reachable Tube (Interactive)"
    
    viz = InteractiveVisualizer(sliders, update_visualization, title=title, direct_plot=True)
    
    print(f"\n{'='*60}")
    print("Interactive Mode")
    print('='*60)
    print(f"State space: {vf.state_dim}D")
    print(f"Time range: [{time_points[0]:.2f}, {time_points[-1]:.2f}] s")
    if vf.state_dim > 2:
        print(f"Slice dimension: {slice_dim}")
    print("\nAdjust sliders to explore the backward reachable tube.")
    print("The black contour shows the boundary of the reachable set.")
    print("Close the window when done.\n")
    
    viz.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize HJ reachability grid value (value function)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--tag', type=str, required=True,
                       help='GridValue cache tag')
    
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Output directory (default: outputs/visualizations/grid_values/{tag}/{preset})')
    # Backward-compat alias
    parser.add_argument('--output', type=str, default=None,
                       help=argparse.SUPPRESS)

    parser.add_argument('--preset', type=str, default=None,
                       help='Visualization preset name from config/visualizations.yaml')

    parser.add_argument('--interpolate', action='store_true',
                       help='Use nearest-neighbor for off-grid times/values')
    
    parser.add_argument('--time-indices', type=int, nargs='+', default=None,
                       help='Time indices to visualize (default: 0, middle, -1)')
    
    parser.add_argument('--slice-dim', type=int, default=2,
                       help='Dimension to slice for 3D state spaces (default: 2)')
    parser.add_argument('--slice-value', type=float, default=None,
                       help='Value at which to slice (default: middle)')
    
    parser.add_argument('--evolution-frames', type=int, default=10,
                       help='Number of frames for evolution plot (default: 10)')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive visualization with sliders')
    
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
    else:
        # Static mode uses Agg for saving files
        matplotlib.use('Agg')
    
    # Load value function
    print("=" * 60)
    print("GridValue Visualization")
    print("=" * 60)
    
    try:
        vf = load_grid_value_by_tag(args.tag, interpolate=args.interpolate)
        if args.interpolate:
            print(f"Interpolation enabled: GridValue can evaluate at arbitrary resolutions")
        else:
            print(f"Interpolation disabled: GridValue will use cached grid points directly")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return

    # Load system for obstacle rendering using cache metadata
    try:
        meta = get_grid_value_metadata(args.tag)
        sys_name = meta.get('system_name', 'UnknownSystem')
        system = instantiate_system_by_name(sys_name)
    except Exception:
        print(f"Warning: Could not instantiate system from metadata")
        system = None
    
    # Interactive mode
    if args.interactive:
        if system is None:
            print("Error: System must be available for interactive visualization")
            return
        run_interactive(vf, system, args, tag=args.tag)
        return
    
    # Determine preset and output dir
    preset = args.preset or 'default'
    if args.save_dir is None:
        output_dir = Path('outputs') / 'visualizations' / 'grid_values' / f'{args.tag}' / preset
    else:
        output_dir = Path(args.save_dir)

    if args.output is not None and args.save_dir is None:
        # Back-compat: treat --output as --save-dir
        output_dir = Path(args.output)

    if args.preset:
        # Preset-driven visualization using shared loader
        sys_name = meta.get('system_name', 'UnknownSystem')
        preset_node = load_visualization_presets(sys_name, 'default', preset)
        if not isinstance(preset_node, dict) or not preset_node:
            print(f"✗ Preset not found for system={sys_name}, input=default, preset={preset}")
            return
        slices: List[Dict[str, Any]] = preset_node.get('slices', []) or []
        if not slices:
            print("⚠ Warning: Preset contains no slices, using empty list")
        # Iterate slices and generate figures
        for s_cfg in slices:
            dims = s_cfg.get('dims', [0, 1])
            if 'dims' not in s_cfg:
                print(f"⚠ Warning: Slice missing 'dims', using default {dims}")
            fixed: Dict[int, float] = s_cfg.get('fixed', {}) or {}
            times_list = s_cfg.get('times', [float(vf._times[0].item())])
            if 'times' not in s_cfg:
                print(f"⚠ Warning: Slice missing 'times', using default {times_list}")
            title = s_cfg.get('title', None)
            # Determine slice axis and index
            if len(dims) != 2:
                print(f"Skipping slice with non-2D dims: {dims}")
                continue
            all_dims = set(range(vf.state_dim))
            fixed_dims = list(all_dims - set(dims))
            if len(fixed_dims) == 1:
                sd = fixed_dims[0]
                if sd < 0 or sd >= vf.state_dim:
                    print(f"Invalid fixed dim: {sd}")
                    continue
                # choose index nearest to fixed value if provided, else center
                axis_t = vf._axes[sd]
                if sd in fixed:
                    val = float(fixed[sd])
                    idx = int(nearest_axis_indices(axis_t, torch.tensor([val], dtype=axis_t.dtype, device=axis_t.device))[0].item())
                else:
                    idx = axis_t.numel() // 2
                slice_dim = sd
                slice_value = float(axis_t[idx].item())
            else:
                # Either 0 or >1 fixed dims; fall back to middle slice heuristics
                slice_dim = 2 if vf.state_dim > 2 else (vf.state_dim - 1)
                axis_t = vf._axes[slice_dim]
                idx = axis_t.numel() // 2
                slice_value = float(axis_t[idx].item())

            # Map times to indices using nearest if interpolate, else require exact
            t_array = vf._times.detach().cpu().numpy()
            time_indices: List[int] = []
            for tv in times_list:
                # Use utility to select nearest index (handles scalar and array)
                cand = int(nearest_time_index(vf._times, float(tv))[0].item())
                if not args.interpolate and abs(float(vf._times[cand].item()) - float(tv)) > 1e-6:
                    raise ValueError(f"Time {tv} not on grid and --interpolate not set")
                time_indices.append(cand)

            # Render using existing helper
            d_for_slice = slice_dim
            # Prepare directory per slice title to avoid overwrite in loops
            # Use a subdirectory per slice to avoid overwriting files
            if title:
                safe = ''.join(ch if (ch.isalnum() or ch in '-_.') else '_' for ch in str(title))
            else:
                safe = f"dims_{dims[0]}_{dims[1]}_slice_{int(slice_dim)}"
            subdir = output_dir / safe
            subdir.mkdir(parents=True, exist_ok=True)
            visualize_2d_slices(
                vf, system, time_indices, subdir,
                slice_dim=d_for_slice,
                slice_value=slice_value,
            )
    else:
        # Fallback manual visualization similar to previous default
        # Determine time indices
        if args.time_indices is None:
            tN = int(vf._times.shape[0]) if getattr(vf, '_times', None) is not None else 1
            mid = max(0, (tN // 2))
            time_indices = [0, mid, -1]
        else:
            time_indices = args.time_indices

        # Generate visualizations
        visualize_2d_slices(
            vf, system, time_indices, output_dir,
            slice_dim=args.slice_dim,
            slice_value=args.slice_value
        )
        visualize_reachable_set_evolution(
            vf, output_dir,
            slice_dim=args.slice_dim,
            slice_value=args.slice_value,
            num_frames=args.evolution_frames
        )
    
    print("\n" + "=" * 60)
    print("✓ Visualization Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
