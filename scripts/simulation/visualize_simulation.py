#!/usr/bin/env python3
"""
Visualize saved simulation results.

Creates video from a saved simulation result pickle file.

The video is saved at real-time speed (FPS = 1/dt) by default. You can adjust
playback speed with --speed and/or resample FPS with --fps.

Usage:
    # Create real-time video
    python scripts/simulation/visualize_simulation.py --tag my_simulation

    # Interactive visualization with time slider
    python scripts/simulation/visualize_simulation.py --tag my_simulation --interactive

    # 2x speed playback
    python scripts/simulation/visualize_simulation.py --tag my_simulation --speed 2.0

    # Resample to 30 FPS (for compatibility)
    python scripts/simulation/visualize_simulation.py --tag my_simulation --fps 30

    # 2x speed AND 30 FPS
    python scripts/simulation/visualize_simulation.py --tag my_simulation --speed 2.0 --fps 30

    # Force overwrite existing outputs
    python scripts/simulation/visualize_simulation.py --tag my_simulation --force
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.utils.registry import instantiate_system
from src.utils.cache_loaders import load_grid_value_by_tag
from src.utils.interactive_viz import InteractiveVisualizer, SliderSpec

# Color constants (keep in sync with compare_values.py)
COLLISION_BLUE = '#0d47a1'  # darker blue for trajectories and markers


def _patch_hide_est(system):
    """Monkey-patch system.render to hide estimated state overlays."""
    import types
    _orig_render = system.render

    def _wrapped_render(self, *args, **kwargs):
        artists = _orig_render(*args, **kwargs)
        try:
            if isinstance(artists, dict):
                if 'est_line' in artists and artists['est_line'] is not None:
                    artists['est_line'].set_visible(False)
                if 'est_heading' in artists and artists['est_heading'] is not None:
                    artists['est_heading'].set_visible(False)
        except Exception:
            pass
        return artists

    system.render = types.MethodType(_wrapped_render, system)


def _apply_same_color():
    """Set matplotlib to use a single color for all lines."""
    import matplotlib
    from matplotlib import cycler
    try:
        first = matplotlib.rcParams.get('axes.prop_cycle').by_key().get('color', ['tab:blue'])[0]
    except Exception:
        first = 'tab:blue'
    matplotlib.rcParams['axes.prop_cycle'] = cycler(color=[first])


def _load_simulation_data(tag: str) -> tuple[dict, 'System'] | tuple[None, None]:
    """Load simulation results and instantiate system.
    
    Returns:
        (data, system) on success, (None, None) on failure (prints error message)
    """
    from src.core.systems import System
    result_path = Path('outputs') / 'simulations' / tag / 'results.pkl'
    
    if not result_path.exists():
        print(f"\n✗ Simulation result not found: {tag}")
        print(f"  Expected path: {result_path}")
        return None, None
    
    print(f"\nLoading simulation: {tag}")
    print(f"  Path: {result_path}")
    
    with open(result_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\nReconstructing simulation results...")
    
    try:
        system = instantiate_system(data['system_name'])
    except ValueError:
        print(f"✗ System class not found: {data['system_name']}")
        return None, None
    
    return data, system


def run_interactive(tag: str, *, same_color: bool = False, hide_est: bool = False, dpi: int = 150, xlim: Optional[tuple[float, float]] = None, ylim: Optional[tuple[float, float]] = None):
    """Run interactive visualization with time slider."""
    data, system = _load_simulation_data(tag)
    if data is None:
        return

    if same_color:
        _apply_same_color()

    if hide_est:
        _patch_hide_est(system)
    
    # Import SimulationResult
    from src.core.simulators import SimulationResult
    
    n_trajectories = data['n_trajectories']
    states = data['states']  # [n_trajectories, time_steps+1, state_dim]
    controls = data['controls']  # [n_trajectories, time_steps, control_dim]
    disturbances = data['disturbances']  # [n_trajectories, time_steps, disturbance_dim]
    uncertainties = data['uncertainties']  # [n_trajectories, time_steps, uncertainty_dim]
    estimated_states = data['estimated_states']  # [n_trajectories, time_steps, state_dim]
    times = data['times']  # [time_steps+1]
    
    results = []
    for i in range(n_trajectories):
        res = SimulationResult()
        res.system = system
        res.system_name = data['system_name']
        res.states = states[i:i+1]  # [1, time_steps+1, state_dim]
        res.controls = controls[i:i+1]  # [1, time_steps, control_dim]
        res.disturbances = disturbances[i:i+1]  # [1, time_steps, disturbance_dim]
        res.uncertainties = uncertainties[i:i+1]  # [1, time_steps, uncertainty_dim]
        res.estimated_states = estimated_states[i:i+1]  # [1, time_steps, state_dim]
        res.times = times
        results.append(res)
    
    print(f"✓ Reconstructed {len(results)} trajectory/trajectories")
    print(f"  Time steps: {len(times)}")
    print(f"  Time range: [{float(times[0]):.3f}, {float(times[-1]):.3f}]")
    
    # Setup interactive visualization
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    try:
        matplotlib.rcParams['figure.dpi'] = dpi
    except Exception:
        pass
    
    num_frames = len(times)
    
    # Store renderer state
    renderer_state = {
        'system': system,
        'results': results,
        'times': times,
        'artists': [None] * len(results)  # Track artists for each trajectory
    }
    
    def visualize_frame(frame_idx, ax=None):
        """Render a single frame of the simulation."""
        if ax is None:
            return

        # Convert to integer (slider values come as floats)
        frame_idx = int(frame_idx)

        ax.clear()

        # Clear the render cache for this axis so it reinitializes
        if hasattr(system, '_render_cache'):
            ax_id = id(ax)
            if ax_id in system._render_cache:
                del system._render_cache[ax_id]

        # Get time for this frame
        time_value = float(times[frame_idx].item())

        # Render each trajectory
        for i, result in enumerate(results):
            # Get state, control, etc. at this frame
            state = _slice_at(result.states, frame_idx)
            control = _slice_at(result.controls, min(frame_idx, result.controls.shape[-2] - 1))
            disturbance = _slice_at(result.disturbances, min(frame_idx, result.disturbances.shape[-2] - 1))
            uncertainty = _slice_at(result.uncertainties, min(frame_idx, result.uncertainties.shape[-2] - 1))

            # Get history up to this frame
            history = _get_history(result, frame_idx)

            # Render this trajectory (pass None for artists since we cleared the axis)
            artists = system.render(
                state=state,
                control=control,
                disturbance=disturbance,
                uncertainty=uncertainty,
                time=time_value,
                ax=ax,
                artists=None,  # Always pass None after clearing
                history=history,
                frame=frame_idx
            )
            renderer_state['artists'][i] = artists
            # Thinner trajectory lines
            try:
                if isinstance(artists, dict):
                    # Force consistent dark-blue coloring to match compare_values.py
                    if 'line' in artists and artists['line'] is not None:
                        artists['line'].set_color(COLLISION_BLUE)
                        artists['line'].set_linewidth(1.2)
                    if 'est_line' in artists and artists['est_line'] is not None:
                        artists['est_line'].set_color(COLLISION_BLUE)
                        artists['est_line'].set_linewidth(1.0)
                        artists['est_line'].set_linestyle(':')
            except Exception:
                pass

        # Draw persistent initial state markers (all trajectories) after trajectories
        try:
            init_states = data.get('initial_states', None)
            if isinstance(init_states, torch.Tensor):
                init_np = init_states.detach().cpu().numpy()
            else:
                init_np = np.asarray(init_states) if init_states is not None else None
            if init_np is not None and init_np.ndim == 2 and init_np.shape[1] >= 2:
                # Determine colors: extract from each trajectory line artist or default
                colors = []
                for art in renderer_state['artists']:
                    if isinstance(art, dict) and 'line' in art and art['line'] is not None:
                        colors.append(art['line'].get_color())
                if len(colors) != init_np.shape[0]:
                    colors = [colors[0] if colors else 'k'] * init_np.shape[0]
                for (x0, y0), c in zip(init_np[:, :2], colors):
                    ax.plot(x0, y0, marker='o', markersize=4, color=c, alpha=0.9)
        except Exception:
            pass

        # Apply axis limits if provided (after rendering, before legend)
        try:
            if xlim is not None and len(xlim) == 2:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None and len(ylim) == 2:
                ax.set_ylim(ylim[0], ylim[1])
        except Exception:
            pass

        # Move legend outside
        try:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                n_items = len(labels)
                ncol = 1 if n_items <= 10 else (2 if n_items <= 20 else 3)
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, ncol=ncol)
        except Exception:
            pass

        # Set title with time information
        ax.set_title(f"Simulation: {tag} | Time: {time_value:.3f}s | Frame: {frame_idx+1}/{num_frames}")
    
    # Create sliders
    sliders = [
        SliderSpec(
            name='frame',
            min_val=0,
            max_val=num_frames - 1,
            initial_val=0,
            step=1,
            description='Time Step'
        )
    ]
    
    # Create interactive visualizer
    viz = InteractiveVisualizer(
        sliders=sliders,
        update_function=visualize_frame,
        direct_plot=True,
        figsize=(10, 8)
    )
    
    viz.show()


def _slice_at(tensor: torch.Tensor, index: int):
    """Extract a slice from a tensor at the given index."""
    if tensor.numel() == 0:
        return None
    if tensor.ndim == 3:
        index = max(min(index, tensor.shape[1] - 1), 0)
        return tensor[0, index]
    index = max(min(index, tensor.shape[0] - 1), 0)
    return tensor[index]


def _get_history(result, frame: int):
    """Get trajectory history up to the given frame."""
    states = result.states
    if states.numel() == 0:
        return None
    # Collapse batch dimension if present
    if states.ndim == 3:
        states = states[0]
    actual_hist = states[: frame + 1]
    # If estimated states are available, include them separately
    est = getattr(result, 'estimated_states', None)
    if est is not None and isinstance(est, torch.Tensor) and est.numel() > 0:
        if est.ndim == 3:
            est = est[0]
        est_hist = est[: frame + 1]
        return {'actual': actual_hist, 'estimated': est_hist}
    return actual_hist


def visualize_simulation(
    tag: str,
    fps: int = None,
    speed: float = 1.0,
    no_video: bool = False,
    force: bool = False,
    *,
    same_color: bool = False,
    hide_est: bool = False,
    dpi: int = 150,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
):
    """Generate video from saved simulation results.
    
    Args:
        tag: Simulation tag to visualize
        fps: Optional target FPS for resampling the video (default: None, keep real-time FPS)
        speed: Playback speed multiplier (default: 1.0 for real-time, 2.0 for 2x, etc.)
        no_video: Skip video generation
        force: Overwrite existing video
    """
    
    data, system = _load_simulation_data(tag)
    if data is None:
        return
    
    # Check if video output already exists
    out_dir = Path('outputs') / 'simulations' / tag
    video_path = out_dir / 'simulation.mp4'
    
    if not force and not no_video and video_path.exists():
        print(f"\n⚠ Video already exists: {video_path}")
        print("  Use --force to overwrite")
        return

    # Styling options for static rendering path
    if same_color:
        _apply_same_color()

    if hide_est:
        _patch_hide_est(system)
    
    # Import SimulationResult and create result objects from batched data
    from src.core.simulators import SimulationResult
    
    n_trajectories = data['n_trajectories']
    states = data['states']  # [n_trajectories, time_steps+1, state_dim]
    controls = data['controls']  # [n_trajectories, time_steps, control_dim]
    disturbances = data['disturbances']  # [n_trajectories, time_steps, disturbance_dim]
    uncertainties = data['uncertainties']  # [n_trajectories, time_steps, uncertainty_dim]
    estimated_states = data['estimated_states']  # [n_trajectories, time_steps, state_dim]
    times = data['times']  # [time_steps+1]
    
    results = []
    for i in range(n_trajectories):
        res = SimulationResult()
        res.system = system
        res.system_name = data['system_name']
        # Extract individual trajectory but keep batch dimension for consistency with SimulationResult
        res.states = states[i:i+1]  # [1, time_steps+1, state_dim]
        res.controls = controls[i:i+1]  # [1, time_steps, control_dim]
        res.disturbances = disturbances[i:i+1]  # [1, time_steps, disturbance_dim]
        res.uncertainties = uncertainties[i:i+1]  # [1, time_steps, uncertainty_dim]
        res.estimated_states = estimated_states[i:i+1]  # [1, time_steps, state_dim]
        res.times = times  # Shared across all trajectories
        results.append(res)
    
    print(f"✓ Reconstructed {len(results)} trajectory/trajectories")
    
    # Create renderer
    import matplotlib
    matplotlib.use('Agg')
    try:
        matplotlib.rcParams['figure.dpi'] = dpi
    except Exception:
        pass
    from src.utils.renderer import SimulationRenderer
    
    dt = float(data['dt'])
    steps = int(data['steps'])
    
    renderer = SimulationRenderer(system, dt=dt, nt=steps + 1)
    try:
        renderer.fig.set_dpi(dpi)
    except Exception:
        pass
    for res in results:
        renderer.add(res)
    # Thinner trajectory lines (after artists are created at add time)
    try:
        for entry in getattr(renderer, '_trajectories', []):
            arts = entry.get('artists') if isinstance(entry, dict) else None
            if isinstance(arts, dict):
                if 'line' in arts and arts['line'] is not None:
                    arts['line'].set_color(COLLISION_BLUE)
                    arts['line'].set_linewidth(1.2)
                if 'est_line' in arts and arts['est_line'] is not None:
                    arts['est_line'].set_color(COLLISION_BLUE)
                    arts['est_line'].set_linewidth(1.0)
                    arts['est_line'].set_linestyle(':')
    except Exception:
        pass
    
    # Calculate real-time FPS
    real_time_fps = 1.0 / dt
    
    print(f"\n✓ Renderer initialized with {len(results)} trajectory/trajectories")
    print(f"  Simulation dt: {dt}s per frame")
    print(f"  Total simulation time: {dt * steps:.2f}s")
    print(f"  Real-time FPS: {real_time_fps:.2f}")
    
    # Add persistent initial state markers before animation (colors from trajectory line artists)
    try:
        init_states = data.get('initial_states', None)
        if isinstance(init_states, torch.Tensor):
            init_np = init_states.detach().cpu().numpy()
        else:
            init_np = np.asarray(init_states) if init_states is not None else None
        if init_np is not None and init_np.ndim == 2 and init_np.shape[1] >= 2:
            colors = []
            for entry in renderer._trajectories:  # type: ignore[attr-defined]
                arts = entry.get('artists')
                if isinstance(arts, dict) and 'line' in arts and arts['line'] is not None:
                    colors.append(arts['line'].get_color())
            if len(colors) != init_np.shape[0]:
                colors = [colors[0] if colors else 'k'] * init_np.shape[0]
            for (x0, y0), c in zip(init_np[:, :2], colors):
                renderer.ax.plot(x0, y0, marker='o', markersize=3, color=c, alpha=0.9)
    except Exception:
        pass

    # Apply axis limits if provided before legend positioning
    try:
        if xlim is not None and len(xlim) == 2:
            renderer.ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None and len(ylim) == 2:
            renderer.ax.set_ylim(ylim[0], ylim[1])
    except Exception:
        pass

    # Reposition legend outside plotting area
    try:
        handles, labels = renderer.ax.get_legend_handles_labels()
        if handles and labels:
            n_items = len(labels)
            ncol = 1 if n_items <= 10 else (2 if n_items <= 20 else 3)
            renderer.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, ncol=ncol)
            renderer.fig.subplots_adjust(right=0.68)
    except Exception:
        pass

    # Save video at real-time FPS
    if not no_video:
        temp_video_path = video_path.with_suffix('.temp.mp4')
        print(f"\nCreating real-time video...")
        renderer.save(str(temp_video_path), fps=int(round(real_time_fps)))
        print(f"✓ Real-time video saved")
        
        # Apply speed adjustment and/or FPS resampling if needed
        needs_processing = (fps is not None) or (speed != 1.0)
        
        if needs_processing:
            print(f"\nProcessing video with ffmpeg...")
            import subprocess
            
            # Build ffmpeg filter
            filters = []
            
            # Speed adjustment using setpts (affects playback speed)
            if speed != 1.0:
                # setpts multiplier is inverse of speed (0.5x speed = 2.0x setpts)
                setpts_value = 1.0 / speed
                filters.append(f"setpts={setpts_value}*PTS")
                print(f"  Applying {speed}x playback speed")
            
            # FPS adjustment
            target_fps = fps if fps is not None else int(round(real_time_fps * speed))
            
            if fps is not None:
                if fps > real_time_fps * speed:
                    print(f"  ⚠ Warning: Target FPS ({fps}) is higher than speed-adjusted real-time FPS ({real_time_fps * speed:.2f})")
                    print(f"    Cannot upsample - using {target_fps} FPS")
                else:
                    print(f"  Resampling to {fps} FPS")
                    target_fps = fps
            
            filters.append(f"fps={target_fps}")
            
            filter_str = ','.join(filters)
            
            result = subprocess.run([
                'ffmpeg', '-y', '-i', str(temp_video_path),
                '-filter:v', filter_str,
                '-c:v', 'libx264',  # Re-encode with H.264
                '-preset', 'slow',  # Slower = better compression at same quality
                '-crf', '18',  # High quality (18 is visually lossless, 0 is lossless)
                '-pix_fmt', 'yuv420p',  # Ensure compatibility
                str(video_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                temp_video_path.unlink()
                print(f"✓ Video processed successfully")
            else:
                print(f"✗ FFmpeg failed: {result.stderr}")
                print(f"  Keeping original video")
                temp_video_path.rename(video_path)
        else:
            # No processing needed, just rename
            temp_video_path.rename(video_path)
        
        print(f"\n✓ Visualization complete")
        print(f"  Video:  {video_path}")
        if speed != 1.0:
            print(f"  Playback speed: {speed}x")
        print(f"  Final FPS: {target_fps if needs_processing else int(round(real_time_fps))}")
    else:
        print(f"\n✓ Visualization complete (video generation skipped)")


def save_final_frame(tag: str, *, out_name: str = 'final_frame.png', same_color: bool = False, hide_est: bool = False, dpi: int = 150, xlim: Optional[tuple[float, float]] = None, ylim: Optional[tuple[float, float]] = None) -> None:
    """Quickly render and save only the final frame to a PNG.

    Uses SimulationRenderer to render all trajectories at the last time step.
    """
    data, system = _load_simulation_data(tag)
    if data is None:
        return

    # Apply styling options
    if same_color:
        _apply_same_color()

    if hide_est:
        _patch_hide_est(system)

    from src.core.simulators import SimulationResult
    states = data['states']
    controls = data['controls']
    disturbances = data['disturbances']
    uncertainties = data['uncertainties']
    estimated_states = data['estimated_states']
    times = data['times']
    dt = float(data['dt'])
    steps = int(data['steps'])
    n_traj = int(data['n_trajectories'])

    import matplotlib
    matplotlib.use('Agg')
    try:
        matplotlib.rcParams['figure.dpi'] = dpi
    except Exception:
        pass
    from src.utils.renderer import SimulationRenderer

    renderer = SimulationRenderer(system, dt=dt, nt=steps + 1)
    try:
        renderer.fig.set_dpi(dpi)
    except Exception:
        pass
    for i in range(n_traj):
        res = SimulationResult()
        res.system = system
        res.system_name = data['system_name']
        res.states = states[i:i+1]
        res.controls = controls[i:i+1]
        res.disturbances = disturbances[i:i+1]
        res.uncertainties = uncertainties[i:i+1]
        res.estimated_states = estimated_states[i:i+1]
        res.times = times
        renderer.add(res)

    # Initialize and update to final frame, then save
    renderer._init_animation()  # type: ignore[attr-defined]
    # Apply thinner lines after artists are initialized
    try:
        for entry in getattr(renderer, '_trajectories', []):
            arts = entry.get('artists') if isinstance(entry, dict) else None
            if isinstance(arts, dict):
                if 'line' in arts and arts['line'] is not None:
                    arts['line'].set_color(COLLISION_BLUE)
                    arts['line'].set_linewidth(1.2)
                if 'est_line' in arts and arts['est_line'] is not None:
                    arts['est_line'].set_color(COLLISION_BLUE)
                    arts['est_line'].set_linewidth(1.0)
                    arts['est_line'].set_linestyle(':')
    except Exception:
        pass
    renderer._update(steps)  # type: ignore[attr-defined]

    # Axis limits if provided
    try:
        if xlim is not None and len(xlim) == 2:
            renderer.ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None and len(ylim) == 2:
            renderer.ax.set_ylim(ylim[0], ylim[1])
    except Exception:
        pass

    # Add initial state markers
    try:
        init_states = data.get('initial_states', None)
        if isinstance(init_states, torch.Tensor):
            init_np = init_states.detach().cpu().numpy()
        else:
            init_np = np.asarray(init_states) if init_states is not None else None
        if init_np is not None and init_np.ndim == 2 and init_np.shape[1] >= 2:
            colors = []
            for entry in renderer._trajectories:  # type: ignore[attr-defined]
                arts = entry.get('artists')
                if isinstance(arts, dict) and 'line' in arts and arts['line'] is not None:
                    colors.append(arts['line'].get_color())
            if len(colors) != init_np.shape[0]:
                colors = [colors[0] if colors else 'k'] * init_np.shape[0]
            for (x0, y0), c in zip(init_np[:, :2], colors):
                renderer.ax.plot(x0, y0, marker='o', markersize=4, color=c, alpha=0.9)
    except Exception:
        pass

    # Overlay light-on points (for systems with a light switch control)
    try:
        # Heuristic: look for a control channel labeled like 'light' or a known Light system
        ctrl_labels = tuple(getattr(system, 'control_labels', ()))
        has_light_label = any('light' in str(lbl).lower() for lbl in ctrl_labels)
        is_light_system = 'light' in system.__class__.__name__.lower()
        if has_light_label or is_light_system:
            # Assume the last control dim corresponds to the light switch when present
            # Prefer an index inferred from labels if available
            light_idx = None
            for idx, lbl in enumerate(ctrl_labels):
                if 'light' in str(lbl).lower():
                    light_idx = idx
                    break
            if light_idx is None:
                light_idx = max(0, int(getattr(system, 'control_dim', 1)) - 1)

            # Plot gold dots where light is ON (>= 0.5) using the state at the same step index
            any_dots = False
            for i in range(n_traj):
                ctrl_i = controls[i]  # [T, U]
                st_i = states[i]      # [T+1, D]
                if ctrl_i.ndim != 2 or st_i.ndim != 2:
                    continue
                if ctrl_i.shape[0] < 1 or st_i.shape[0] < 1:
                    continue
                light_series = ctrl_i[:, light_idx].detach().cpu().numpy()
                on_idx = np.where(light_series >= 0.5)[0]
                if on_idx.size == 0:
                    continue
                # Clip to valid state indices (use state at the same step; renderer uses state[frame])
                on_idx = on_idx[on_idx < st_i.shape[0]]
                if on_idx.size == 0:
                    continue
                pts = st_i[on_idx, :2].detach().cpu().numpy()
                renderer.ax.plot(pts[:, 0], pts[:, 1], 'o', color='gold', markersize=3, label='light on' if not any_dots else None, zorder=15)
                any_dots = True
    except Exception:
        pass

    # Legend outside
    try:
        handles, labels = renderer.ax.get_legend_handles_labels()
        if handles and labels:
            n_items = len(labels)
            ncol = 1 if n_items <= 10 else (2 if n_items <= 20 else 3)
            renderer.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, ncol=ncol)
            renderer.fig.subplots_adjust(right=0.68)
    except Exception:
        pass
    out_dir = Path('outputs') / 'simulations' / tag
    final_path = out_dir / out_name
    renderer.fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved final frame: {final_path}")


def visualize_value_zero_level(
    tag: str,
    value_name: str,
    value_tag: str | None,
    value_time: float = 0.0,
    dpi: int = 150,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """Plot the zero-level set of a specified Value (e.g., GridValue).

    Prefers using the simulation's saved grid slice metadata to define the evaluation plane.
    If unavailable, falls back to the Value's native grid (first two dimensions).
    """
    data, system = _load_simulation_data(tag)
    if data is None:
        return

    # Resolve and load the Value
    if value_name != 'GridValue':
        print(f"✗ Unsupported --value '{value_name}'. Supported: GridValue")
        return
    if not value_tag:
        print("✗ --value-tag is required when --value=GridValue")
        return
    gv = load_grid_value_by_tag(value_tag, interpolate=True)

    # Prefer the simulation grid metadata if present
    grid_meta = data.get('initial_states_grid_meta', None)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path('outputs') / 'simulations' / tag
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = getattr(system, 'state_labels', tuple([f'x{i}' for i in range(getattr(system, 'state_dim', 2))]))

    if isinstance(grid_meta, dict) and grid_meta.get('type') == 'grid_slice_2d':
        nx, ny = int(grid_meta['resolution'][0]), int(grid_meta['resolution'][1])
        v0, v1 = int(grid_meta['vary_dims'][0]), int(grid_meta['vary_dims'][1])
        lo0, lo1 = float(grid_meta['limits']['lo'][0]), float(grid_meta['limits']['lo'][1])
        hi0, hi1 = float(grid_meta['limits']['hi'][0]), float(grid_meta['limits']['hi'][1])
        fixed = grid_meta.get('grid_fixed_values') or grid_meta.get('fixed_values')
        if not fixed:
            fixed = data['initial_states'][0].tolist()

        x = np.linspace(lo0, hi0, nx)
        y = np.linspace(lo1, hi1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        # Build query states
        state_dim = int(system.state_dim)
        pts = torch.as_tensor(fixed, dtype=torch.float32).repeat(nx * ny, 1)
        pts[:, v0] = torch.as_tensor(X.reshape(-1), dtype=torch.float32)
        pts[:, v1] = torch.as_tensor(Y.reshape(-1), dtype=torch.float32)
        with torch.no_grad():
            V = gv.value(pts, value_time).reshape(nx, ny).cpu().numpy()
        CS = ax.contour(X, Y, V, levels=[0.0], colors='k', linewidths=2)
        ax.set_xlabel(labels[v0] if v0 < len(labels) else f'x{v0}')
        ax.set_ylabel(labels[v1] if v1 < len(labels) else f'x{v1}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.set_title(f"Zero-level set of {value_name} (tag={value_tag}, t={value_time:g})")
        # Axis limits
        try:
            if xlim is not None and len(xlim) == 2:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None and len(ylim) == 2:
                ax.set_ylim(ylim[0], ylim[1])
        except Exception:
            pass
    else:
        # Fallback: use the value's native grid (first two dimensions)
        axes = gv.metadata.get('grid_coordinate_vectors')
        if axes is None or len(axes) < 2:
            print("✗ Cannot infer a 2D slice to plot zero-level set")
            return
        x = np.asarray(axes[0])
        y = np.asarray(axes[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        # Build query states at the native grid points (first two dims vary, others fixed to midpoints)
        state_dim = int(system.state_dim)
        mids = []
        for axis in axes:
            arr = np.asarray(axis)
            mids.append(float(arr[len(arr)//2]))
        base = torch.as_tensor(mids[:state_dim], dtype=torch.float32)
        pts = base.repeat(X.size, 1)
        pts[:, 0] = torch.as_tensor(X.reshape(-1), dtype=torch.float32)
        pts[:, 1] = torch.as_tensor(Y.reshape(-1), dtype=torch.float32)
        with torch.no_grad():
            V = gv.value(pts, value_time).reshape(X.shape[0], X.shape[1]).cpu().numpy()
        CS = ax.contour(X, Y, V, levels=[0.0], colors='k', linewidths=2)
        ax.set_xlabel(labels[0] if len(labels) > 0 else 'x0')
        ax.set_ylabel(labels[1] if len(labels) > 1 else 'x1')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.set_title(f"Zero-level set of {value_name} (tag={value_tag}, t={value_time:g})")
        # Axis limits
        try:
            if xlim is not None and len(xlim) == 2:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None and len(ylim) == 2:
                ax.set_ylim(ylim[0], ylim[1])
        except Exception:
            pass

    out_path = out_dir / 'value_zero_level.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"✓ Saved zero-level set: {out_path}")


def visualize_min_failure(tag: str, *, show: bool = False, save: bool = True, dpi: int = 150, xlim: Optional[tuple[float, float]] = None, ylim: Optional[tuple[float, float]] = None) -> None:
    """Plot the minimum failure value along each trajectory's time history.

    If the simulation was initialized from a 2D grid slice (saved metadata present),
    render a heatmap/contour over that grid. Otherwise, produce a scatter plot over
    two chosen dimensions (prefer vary_dims from metadata if available, else [0,1]).
    Always saves a figure; no text outputs.
    """
    if show and not save:
        # Use interactive backend for showing a window
        import matplotlib
        matplotlib.use('TkAgg')
    else:
        import matplotlib
        matplotlib.use('Agg')

    data, system = _load_simulation_data(tag)
    if data is None:
        return

    states = data['states']  # [N, T+1, D]
    initial_states = data['initial_states']  # [N, D]
    N, T1, D = int(states.shape[0]), int(states.shape[1]), int(states.shape[2])

    # Compute failure over all time and take min per trajectory
    use_gpu = getattr(system, '_use_gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    bs = int(getattr(system, '_batch_size', 100000))
    print(f"Computing failure values: device={device}, batch={bs}, trajectories={N}, steps={T1}")

    flat = states.reshape(N * T1, D)
    mins = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], bs):
            chunk = flat[i:i+bs].to(device)
            vals = system.failure_function(chunk, None)
            mins.append(vals.detach().cpu())
    vals_all = torch.cat(mins, dim=0).reshape(N, T1)
    min_per_traj = torch.min(vals_all, dim=1).values  # [N]

    out_dir = Path('outputs') / 'simulations' / tag
    grid_meta = data.get('initial_states_grid_meta', None)

    import matplotlib.pyplot as plt
    labels = getattr(system, 'state_labels', tuple([f'x{i}' for i in range(D)]))

    def _render_heatmap(X, Y, Z, v0, v1):
        # Align color mapping with GridValue viz: symmetric, centered at 0
        from matplotlib.colors import TwoSlopeNorm
        vabs = max(abs(float(np.min(Z))), abs(float(np.max(Z))))
        if vabs == 0:
            vabs = 1.0
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        levels = np.linspace(-vabs, vabs, 21)
        fig, ax = plt.subplots(figsize=(8, 7))
        cf = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlBu', norm=norm)
        # Zero level contour
        try:
            ax.contour(X, Y, Z, levels=[0.0], colors='black', linewidths=2, linestyles='-')
        except Exception:
            pass
        # Obstacles (match visualize_grid_value)
        if D >= 2 and v0 >= 0 and v1 >= 0:
            try:
                from src.utils.obstacles import draw_obstacles_2d
                draw_obstacles_2d(ax, system, zorder=10)
            except Exception:
                pass
        ax.set_xlabel(labels[v0] if v0 < len(labels) else f'x{v0}')
        ax.set_ylabel(labels[v1] if v1 < len(labels) else f'x{v1}')
        ax.set_title(f"Min failure over trajectories (tag={tag})")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu')
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='min failure')
        # Axis limits
        try:
            if xlim is not None and len(xlim) == 2:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None and len(ylim) == 2:
                ax.set_ylim(ylim[0], ylim[1])
        except Exception:
            pass
        fig.tight_layout()
        if save:
            out_path = out_dir / 'min_failure.png'
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved min-failure heatmap: {out_path}")
        if show:
            plt.show()
        plt.close(fig)

    if isinstance(grid_meta, dict) and grid_meta.get('type') == 'grid_slice_2d':
        # Heatmap/contour over the configured slice
        nx, ny = int(grid_meta.get('resolution', [0, 0])[0]), int(grid_meta.get('resolution', [0, 0])[1])
        v0, v1 = int(grid_meta.get('vary_dims', [0, 1])[0]), int(grid_meta.get('vary_dims', [0, 1])[1])
        limits = grid_meta.get('limits', {})
        lo_list = limits.get('lo', [float(initial_states[:, v0].min()), float(initial_states[:, v1].min())])
        hi_list = limits.get('hi', [float(initial_states[:, v0].max()), float(initial_states[:, v1].max())])
        lo0, lo1 = float(lo_list[0]), float(lo_list[1])
        hi0, hi1 = float(hi_list[0]), float(hi_list[1])

        # If resolution isn't set or mismatched, attempt inference
        if nx * ny != N:
            nx = int(np.sqrt(N))
            ny = int(np.ceil(N / max(nx, 1)))

        try:
            Z = min_per_traj.reshape(nx, ny).numpy()
            x = np.linspace(lo0, hi0, nx)
            y = np.linspace(lo1, hi1, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            _render_heatmap(X, Y, Z, v0, v1)
            return
        except Exception:
            pass  # Fall through to inference/scatter

    # Attempt to infer a grid from initial_states if metadata missing
    inferred = False
    best = None
    for v0 in range(min(D, 4)):
        for v1 in range(v0 + 1, min(D, 4)):
            a0 = initial_states[:, v0].detach().cpu().numpy()
            a1 = initial_states[:, v1].detach().cpu().numpy()
            u0, inv0 = np.unique(a0, return_inverse=True)
            u1, inv1 = np.unique(a1, return_inverse=True)
            if u0.size * u1.size != N:
                continue
            # Try to populate a grid
            Z = np.full((u0.size, u1.size), np.nan, dtype=float)
            vals = min_per_traj.detach().cpu().numpy()
            for i in range(N):
                Z[inv0[i], inv1[i]] = vals[i]
            if np.isnan(Z).any():
                continue
            X, Y = np.meshgrid(u0, u1, indexing='ij')
            best = (X, Y, Z, v0, v1)
            inferred = True
            break
        if inferred:
            break
    if inferred and best is not None:
        X, Y, Z, v0, v1 = best
        _render_heatmap(X, Y, Z, v0, v1)
        return

    # Scatter plot fallback (no or invalid grid metadata)
    # Choose dims: prefer vary_dims from meta, else first two dimensions
    if isinstance(grid_meta, dict) and 'vary_dims' in grid_meta:
        v0, v1 = int(grid_meta['vary_dims'][0]), int(grid_meta['vary_dims'][1])
    else:
        v0, v1 = 0, 1 if D >= 2 else (0, 0)

    pts = initial_states[:, [v0, v1]].detach().cpu().numpy()
    vals = min_per_traj.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    # Color mapping aligned with GridValue viz
    from matplotlib.colors import TwoSlopeNorm
    vabs = max(abs(float(np.min(vals))), abs(float(np.max(vals))))
    if vabs == 0:
        vabs = 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap='RdYlBu', norm=norm, s=20, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='min failure')
    # Obstacles overlay if spatial dims selected
    if D >= 2:
        try:
            from src.utils.obstacles import draw_obstacles_2d
            draw_obstacles_2d(ax, system, zorder=10)
        except Exception:
            pass
    ax.set_xlabel(labels[v0] if v0 < len(labels) else f'x{v0}')
    ax.set_ylabel(labels[v1] if v1 < len(labels) else f'x{v1}')
    ax.set_title(f"Min failure over trajectories (tag={tag})")
    ax.grid(True, alpha=0.3)
    # Axis limits
    try:
        if xlim is not None and len(xlim) == 2:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])
    except Exception:
        pass
    fig.tight_layout()
    if save:
        out_path = out_dir / 'min_failure.png'
        fig.savefig(out_path, dpi=dpi)
        print(f"✓ Saved min-failure scatter: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def _is_grid_simulation(tag: str) -> bool:
    """Heuristically determine if the saved simulation was run over a grid.

    Current criterion: results.pkl contains an 'initial_states_grid_meta' dict.
    """
    result_path = Path('outputs') / 'simulations' / tag / 'results.pkl'
    try:
        if not result_path.exists():
            return False
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
        grid_meta = data.get('initial_states_grid_meta', None)
        return isinstance(grid_meta, dict)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--tag', type=str, required=True, help='Simulation tag to visualize')
    parser.add_argument('--interactive', action='store_true', help='Interactive visualization with time slider')
    parser.add_argument('--fps', type=int, default=None, 
                       help='Resample video to this FPS after creation (default: keep real-time FPS = 1/dt). '
                            'Note: Cannot upsample, only downsample.')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed multiplier (default: 1.0 for real-time). '
                            'Use 2.0 for 2x speed, 0.5 for half speed, etc.')
    parser.add_argument('--no-video', action='store_true', help='Skip video generation')
    parser.add_argument('--force', action='store_true', help='Overwrite existing video')
    parser.add_argument('--save-final-frame', action='store_true', help='Save only the final frame PNG and exit')
    parser.add_argument('--same-color', action='store_true', help='Plot all trajectories with the same color')
    parser.add_argument('--hide-est', action='store_true', help='Hide estimated trajectory overlays (dashed gray)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for all generated figures/videos (default: 150)')
    parser.add_argument('--xlim', type=float, nargs=2, metavar=('XMIN','XMAX'), default=None, help='x-axis limits (e.g. --xlim -5 5)')
    parser.add_argument('--ylim', type=float, nargs=2, metavar=('YMIN','YMAX'), default=None, help='y-axis limits (e.g. --ylim -5 5)')
    # --show-heading flag removed
    # Value plotting (zero-level set)
    parser.add_argument('--value', type=str, default=None, help='Value class to plot zero-level set for (e.g., GridValue)')
    parser.add_argument('--value-tag', type=str, default=None, help='Cache tag for the value (required for GridValue)')
    parser.add_argument('--value-time', type=float, default=0.0, help='Time at which to evaluate the value (default: 0.0)')
    parser.add_argument('--value-zero-level', action='store_true', help='Plot the zero-level set of the chosen value')
    # min-failure is now automatic for grid simulations; flag removed
    
    args = parser.parse_args()
    
    if args.save_final_frame:
        save_final_frame(tag=args.tag, same_color=args.same_color, hide_est=args.hide_est, dpi=args.dpi, xlim=tuple(args.xlim) if args.xlim else None, ylim=tuple(args.ylim) if args.ylim else None)
        return
    elif args.value_zero_level and args.value:
        visualize_value_zero_level(tag=args.tag, value_name=args.value, value_tag=args.value_tag, value_time=args.value_time, dpi=args.dpi, xlim=tuple(args.xlim) if args.xlim else None, ylim=tuple(args.ylim) if args.ylim else None)
    else:
        is_grid = _is_grid_simulation(args.tag)
        if is_grid:
            if args.interactive:
                print("Detected grid-based simulation; showing min-failure visualization interactively.")
                visualize_min_failure(tag=args.tag, show=True, save=False, dpi=args.dpi, xlim=tuple(args.xlim) if args.xlim else None, ylim=tuple(args.ylim) if args.ylim else None)
            else:
                print("Detected grid-based simulation; generating min-failure visualization instead of video.")
                visualize_min_failure(tag=args.tag, dpi=args.dpi, xlim=tuple(args.xlim) if args.xlim else None, ylim=tuple(args.ylim) if args.ylim else None)
        else:
            if args.interactive:
                run_interactive(tag=args.tag, same_color=args.same_color, hide_est=args.hide_est, dpi=args.dpi, xlim=tuple(args.xlim) if args.xlim else None, ylim=tuple(args.ylim) if args.ylim else None)
            else:
                visualize_simulation(
                    tag=args.tag,
                    fps=args.fps,
                    speed=args.speed,
                    no_video=args.no_video,
                    force=args.force,
                    same_color=args.same_color,
                    hide_est=args.hide_est,
                    dpi=args.dpi,
                    xlim=tuple(args.xlim) if args.xlim else None,
                    ylim=tuple(args.ylim) if args.ylim else None,
                )


if __name__ == '__main__':
    main()
