#!/usr/bin/env python3
"""
Build a tagged GridValue cache for HJ reachability dynamics.

This script computes value functions for a chosen hj_reachability dynamics implementation,
binding channels (control, disturbance, uncertainty) via cache tags or input classes,
and saves the results at .cache/grid_values/{TAG}.pkl.

Usage:
    # List available dynamics and cache tags
    python scripts/grid_value/build_grid_value.py --list

    # Build with RoverDark dynamics using a prebuilt GridSet tag for control
    python scripts/grid_value/build_grid_value.py \
        --dynamics RoverDark \
        --control-grid-set-tag {GRID_SET_TAG} \
        --tag {TAG} \
        --description "My value function"

Channel binding flags (mutually exclusive per channel):
    --control-grid-set-tag TAG | --control-grid-input-tag TAG | --control-input NAME
    --disturbance-grid-input-tag TAG | --disturbance-input NAME
    --uncertainty-grid-input-tag TAG | --uncertainty-input NAME
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
import jax
import jax.numpy as jnp
import numpy as np
import torch

import hj_reachability as hj

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.systems import System
from src.impl.values.grid_value import GridValue
from src.utils.cache_loaders import (
    get_grid_input_metadata,
    get_grid_set_metadata,
    instantiate_system_by_name,
    resolve_hj_dynamics_class,
    resolve_input,
    resolve_set,
)
from src.utils.config import load_resolution_config
from src.utils.registry import (
    get_available_hj_dynamics_classes,
    get_available_input_classes,
    list_grid_set_tags,
)


def list_available():
    """List available dynamics, input classes, and existing GridSet tags."""
    dyns = sorted(get_available_hj_dynamics_classes(), key=lambda c: c.__name__)
    inputs = [c for c in get_available_input_classes() if c.__name__ not in ('GridInput',)]
    print("\n" + "=" * 60)
    print("HJ Dynamics (solver-compatible)")
    print("=" * 60)
    for dyn in dyns:
        print(f"  - {dyn.__name__}")
    print("\n" + "=" * 60)
    print("Inputs (implementations)")
    print("=" * 60)
    for inp in inputs:
        t = getattr(inp, 'type', 'any')
        print(f"  - {inp.__name__} (type={t})")
    print("\n" + "=" * 60)
    print("Existing GridSet tags")
    print("=" * 60)
    tags = list_grid_set_tags()
    if tags:
        for t in tags:
            print(f"  - {t}")
    else:
        print("  (none)")
    print()


def _require_keys(d: dict, keys):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Grid config missing required keys: {missing}")
    return d


def compute_initial_values(
    grid: Any,
    system: System
) -> jnp.ndarray:
    """Compute initial values using the system's public failure_function.

    The failure_function should return a signed value per state (negative in failure/obstacle region),
    matching the semantics expected by HJ reachability initial conditions.
    
    Args:
        grid: HJ reachability grid
        system: System instance
    """
    
    # Pull computational settings from system
    use_gpu = getattr(system, '_use_gpu', False)
    batch_size = getattr(system, '_batch_size', 10000)
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if use_gpu and cuda_available else 'cpu'

    print("\nComputing initial values via system.failure_function ...")
    
    # Log attribute sources
    if hasattr(system, '_use_gpu'):
        print(f"  use_gpu={use_gpu} (from System._use_gpu attribute)")
    else:
        print(f"  use_gpu={use_gpu} (using default, System has no _use_gpu attribute)")
    
    if hasattr(system, '_batch_size'):
        print(f"  batch_size={batch_size} (from System._batch_size attribute)")
    else:
        print(f"  batch_size={batch_size} (using default, System has no _batch_size attribute)")
    
    if use_gpu and not cuda_available:
        print(f"  Device: {device} (CUDA not available, falling back to cpu)")
    else:
        print(f"  Device: {device}")

    # Flatten grid states for batched evaluation
    states_flat = np.array(grid.states).reshape(-1, grid.states.shape[-1])

    vals: list[np.ndarray] = []
    for i in range(0, states_flat.shape[0], batch_size):
        batch_np = states_flat[i:i + batch_size]
        batch_t = torch.from_numpy(batch_np).float().to(device)
        print(f'Assuming a time-invariant failure function...')
        with torch.no_grad():
            v_t = system.failure_function(batch_t, 0.0)
        vals.append(v_t.detach().cpu().numpy())

    sdf_flat = np.concatenate(vals, axis=0)
    sdf_grid = sdf_flat.reshape(tuple(int(s) for s in grid.states.shape[:-1]))

    print("✓ Initial values computed")
    try:
        print(f"  Min: {float(np.min(sdf_grid)):.3f}")
        print(f"  Max: {float(np.max(sdf_grid)):.3f}")
        print(f"  Failure fraction: {float((sdf_grid < 0).mean()):.2%}")
    except Exception:
        pass

    return jnp.array(sdf_grid)


def solve_hj_reachability(
    dynamics: Any,
    grid: Any,
    initial_values: jnp.ndarray,
    time_horizon: float,
    time_steps: int,
    accuracy: str = 'very_high',
    progress_bar: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve HJ reachability PDE and compute gradients.
    
    Returns:
        values: Value function array [time_steps, *grid_shape]
        times: Time array [time_steps]
        gradients: Gradient array [time_steps, *grid_shape, state_dim]
    """
    
    print(f"\nSolving HJ Reachability:")
    print(f"  Time horizon: {time_horizon:.2f} s")
    print(f"  Time steps: {time_steps}")
    print(f"  Grid shape: {grid.states.shape[:-1]}")
    print(f"  Accuracy: {accuracy}")
    
    # Time vector (backwards from time_horizon to 0)
    times = np.linspace(time_horizon, 0, time_steps)
    
    # Solver settings
    # Use more stable dissipation (LLF) and a slightly smaller CFL to reduce NaN risk
    solver_settings = hj.SolverSettings.with_accuracy(
        accuracy,
        # For safety/BRT, limit Hamiltonian to non-positive to avoid expansion past target
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        # Bound value function: lower bound by min of initial values, upper bound by initial value at current state
        # This ensures V(t,x) stays in [min(V0), V0(x)] which is physically correct for BRT
        value_postprocessor=(lambda t, v: jnp.clip(v, jnp.min(initial_values), None)),
    )
    # Prefer local Lax-Friedrichs for tighter, state-dependent dissipation
    solver_settings = solver_settings.replace(
        artificial_dissipation_scheme=hj.artificial_dissipation.local_lax_friedrichs,
        CFL_number=0.75,  # default (can change to trade off speed and stability)
    )
    
    # Solve!
    print("\nRunning solver...")
    values = hj.solve(
        solver_settings,
        dynamics,
        grid,
        times,
        initial_values,
        progress_bar=progress_bar
    )
    
    print(f"✓ Solver completed")
    
    # Compute gradients using HJ reachability's built-in method
    # Vectorize over time dimension for parallel computation
    print("\nComputing gradients...")
    
    # Use vmap to parallelize gradient computation across time steps
    grad_fn = lambda val_slice: grid.grad_values(val_slice, solver_settings.upwind_scheme)
    gradients = jax.vmap(grad_fn)(values)
    
    print(f"✓ Gradients computed (parallelized via vmap)")
    print(f"  Gradient shape: {gradients.shape}")
    
    return np.array(values), times, np.array(gradients)


def build_value_function(
    dynamics_name: str,
    *,
    system_name: Optional[str] = None,
    system_args: Optional[Dict[str, Any]] = None,
    control_grid_set_tag: Optional[str] = None,
    control_grid_input_tag: Optional[str] = None,
    control_input_name: Optional[str] = None,
    disturbance_grid_input_tag: Optional[str] = None,
    disturbance_input_name: Optional[str] = None,
    uncertainty_grid_input_tag: Optional[str] = None,
    uncertainty_input_name: Optional[str] = None,
    tag: str,
    description: str = '',
    grid_resolution: Optional[Tuple[int, ...]] = None,
    state_bounds: Optional[Tuple[float, ...]] = None,
    time_horizon: Optional[float] = None,
    time_steps: Optional[int] = None,
    accuracy: str = 'very_high',
    config_path: str = 'config/resolutions.yaml',
    progress_bar: bool = True,
) -> GridValue:
    """
    Build GridValue for a system-controller pair.
    
    Args:
        dynamics_name: Name of the HJ dynamics class
        system_name: Explicit system class name (overrides system inferred from GridSet tag).
        system_args: Dict of key=value pairs to set on the system after instantiation.
                     E.g., {'failure_grid_value_tag': 'MyTag', 'uncertainty_growth_rate': 0.5}
        control_grid_set_tag: GridSet tag for control channel (mutually exclusive with others)
        control_grid_input_tag: GridInput tag for control channel (mutually exclusive with others)
        control_input_name: Input class name for control channel (mutually exclusive with others)
        disturbance_grid_input_tag: GridInput tag for disturbance channel
        disturbance_input_name: Input class name for disturbance channel
        uncertainty_grid_input_tag: GridInput tag for uncertainty channel
        uncertainty_input_name: Input class name for uncertainty channel
        tag: Cache tag for output
        description: Optional description
        grid_resolution: Grid shape (default: from config)
        state_bounds: State bounds [lo_1, ..., lo_n, hi_1, ..., hi_n] (default: from system)
        time_steps: Number of time steps (default: from config)
        accuracy: Solver accuracy level
        config_path: Path to grid resolution config
        progress_bar: Show solver progress bar
    
    Returns:
        GridValue instance (also saved to cache)
    """
    
    print(f"Building GridValue tag '{tag}' for dynamics: {dynamics_name}")
    
    # Determine system_name: explicit --system takes precedence over GridSet/GridInput metadata
    resolved_system_name: Optional[str] = system_name
    input_name: Optional[str] = None
    if control_grid_set_tag is not None:
        gs_meta = get_grid_set_metadata(control_grid_set_tag)
        if resolved_system_name is None:
            resolved_system_name = gs_meta.get('system_name')
        input_name = gs_meta.get('input_name')
    if resolved_system_name is None and control_grid_input_tag is not None:
        gi_meta = get_grid_input_metadata(control_grid_input_tag)
        resolved_system_name = gi_meta.get('system_name')
        input_name = gi_meta.get('input_name')
    system_name = resolved_system_name
    # set type is stored with the GridSet, not needed for value cache
    
    # Step 1: Instantiate system and dynamics
    # Instantiate system
    if system_name is None:
        raise ValueError("Cannot resolve system_name. Provide --system, --control-grid-set-tag, or --control-grid-input-tag.")
    system = instantiate_system_by_name(system_name)
    print(f"✓ System instantiated: {system_name}")
    
    # Apply any system-specific arguments
    if system_args:
        for key, value in system_args.items():
            # Try setter method first (e.g., set_failure_grid_value_tag)
            setter = getattr(system, f'set_{key}', None)
            if callable(setter):
                setter(value)
                print(f"  {key} = {value} (via setter)")
            elif hasattr(system, key):
                setattr(system, key, value)
                print(f"  {key} = {value}")
            else:
                raise ValueError(f"System {system_name} has no attribute '{key}'")

    # Create dynamics (per public interface, no required ctor args)
    HJDynamicsClass = resolve_hj_dynamics_class(dynamics_name)
    dynamics = HJDynamicsClass()

    # Bind control channel (Set or Input, mutually exclusive)
    ctrl_set, ctrl_set_name = resolve_set(system, grid_set_tag=control_grid_set_tag)
    ctrl_input, ctrl_input_name_resolved = resolve_input(
        system,
        input_class=control_input_name,
        grid_input_tag=control_grid_input_tag,
        role="control",
    )
    if ctrl_set and ctrl_input:
        raise ValueError("Specify only one of --control-grid-set-tag, --control-grid-input-tag, or --control-input")
    if ctrl_set:
        dynamics.bind_control_set(ctrl_set)
        input_name = input_name or gs_meta.get('input_name', 'UnknownInput')
    elif ctrl_input:
        dynamics.bind_control_input(ctrl_input)
        input_name = ctrl_input_name_resolved

    # Bind disturbance channel (Input only)
    dist_input, _ = resolve_input(
        system,
        input_class=disturbance_input_name,
        grid_input_tag=disturbance_grid_input_tag,
        role="disturbance",
    )
    if dist_input:
        dynamics.bind_disturbance_input(dist_input)

    # Bind uncertainty channel (Input only)
    unc_input, _ = resolve_input(
        system,
        input_class=uncertainty_input_name,
        grid_input_tag=uncertainty_grid_input_tag,
        role="uncertainty",
    )
    if unc_input:
        dynamics.bind_uncertainty_input(unc_input)

    # Validate dynamics configuration
    dynamics.validate()
    
    # Load grid configuration now that system/input are known
    print(f"Loading configuration from: {config_path}")
    grid_config = load_resolution_config(system_name, input_name or 'default', config_path)
    grid_config = _require_keys(grid_config or {}, ['state_resolution', 'time_resolution'])
    
    # Use config values as defaults
    if grid_resolution is None:
        grid_resolution = tuple(grid_config.get('state_resolution', [50] * 3))
        if 'state_resolution' in grid_config:
            print(f"Grid resolution: {grid_resolution} (from config)")
        else:
            print(f"Grid resolution: {grid_resolution} (using fallback default)")
    else:
        print(f"Grid resolution: {grid_resolution} (from CLI)")
    
    if time_horizon is None:
        time_horizon = float(getattr(system, 'time_horizon'))
        print(f"Time horizon: {time_horizon} (from System.time_horizon)")
    else:
        time_horizon = float(time_horizon)
        print(f"Time horizon: {time_horizon} (from CLI --time-horizon)")
    
    if time_steps is None:
        time_steps = grid_config.get('time_resolution', 50)
        if 'time_resolution' in grid_config:
            print(f"Time steps: {time_steps} (from config)")
        else:
            print(f"Time steps: {time_steps} (using fallback default)")
    else:
        print(f"Time steps: {time_steps} (from CLI)")
    
    # Step: Setup reachability grid
    # Setup reachability grid
    
    # Get state bounds from system if not specified
    if state_bounds is None:
        # Try multiple common attribute names for state bounds
        if hasattr(system, 'state_limits'):
            # PyTorch tensor format: [[low...], [high...]]
            state_limits = system.state_limits
            bounds_lo = jnp.array(state_limits[0].numpy())
            bounds_hi = jnp.array(state_limits[1].numpy())
            print(f"State bounds: from System.state_limits attribute")
        elif hasattr(system, 'state_space'):
            # Gym-style state space
            state_space = system.state_space
            bounds_lo = jnp.array(state_space.low.numpy())
            bounds_hi = jnp.array(state_space.high.numpy())
            print(f"State bounds: from System.state_space attribute")
        else:
            raise ValueError(
                f"System {system_name} does not have 'state_limits' or 'state_space' attribute.\n"
                f"Please specify --state-bounds manually."
            )
    else:
        # Parse state bounds
        n = len(state_bounds) // 2
        bounds_lo = jnp.array(state_bounds[:n])
        bounds_hi = jnp.array(state_bounds[n:])
        print(f"State bounds: from CLI (--state-bounds)")
    
    print(f"Grid bounds: {bounds_lo} to {bounds_hi}")
    print(f"Grid shape: {grid_resolution}")
    print(f"Total points: {np.prod(grid_resolution):,}")
    
    # Determine periodic dimensions (system-specific)
    # Try multiple attribute names and convert boolean list to indices if needed
    periodic_dims = None
    if hasattr(system, 'periodic_state_dims'):
        # Direct indices format [2] or [0, 2]
        periodic_dims = system.periodic_state_dims
        print(f"Periodic dimensions: {periodic_dims} (from System.periodic_state_dims attribute)")
    elif hasattr(system, 'state_periodic'):
        # Boolean format [False, False, True] -> convert to indices [2]
        state_periodic = system.state_periodic
        periodic_dims = [i for i, is_periodic in enumerate(state_periodic) if is_periodic]
        if not periodic_dims:  # Empty list means no periodic dims
            periodic_dims = None
        if periodic_dims:
            print(f"Periodic dimensions: {periodic_dims} (from System.state_periodic attribute)")
    
    if periodic_dims is None:
        print(f"Periodic dimensions: None (no periodic states)")
    
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(lo=bounds_lo, hi=bounds_hi),
        shape=grid_resolution,
        periodic_dims=periodic_dims
    )
    
    # Step 5: Compute initial values
    # Compute initial values
    
    initial_values = compute_initial_values(grid, system)
    
    # Step 6: Solve HJ reachability
    # Solve HJ reachability

    # Enforce GPU usage if available
    try:
        gpu_devices = jax.devices("gpu")
    except:
        gpu_devices = jax.devices("METAL")

    # if gpu_devices:
        # print(f"✓ Using GPU: {gpu_devices[0]}")
    # else:
        # print("⚠ No GPU detected by JAX; running on CPU may be slow.")
    
    values, times, gradients = solve_hj_reachability(
        dynamics,
        grid,
        initial_values,
        time_horizon,
        time_steps,
        accuracy,
        progress_bar
    )
    
    # Step 7: Create and save GridValue
    # Save GridValue
    
    # Prepare bindings descriptor for reconstructing dynamics at load-time
    bindings = {
        'control': (
            {'kind': 'set', 'grid_set_tag': control_grid_set_tag} if control_grid_set_tag else
            ({'kind': 'input', 'grid_input_tag': control_grid_input_tag} if control_grid_input_tag else
            ({'kind': 'input', 'input_name': control_input_name} if control_input_name else {'kind': 'optimize'}))
        ),
        'disturbance': (
            {'kind': 'input', 'grid_input_tag': disturbance_grid_input_tag} if disturbance_grid_input_tag else
            ({'kind': 'input', 'input_name': disturbance_input_name} if disturbance_input_name else {'kind': 'zero'})
        ),
        'uncertainty': (
            {'kind': 'input', 'grid_input_tag': uncertainty_grid_input_tag} if uncertainty_grid_input_tag else
            ({'kind': 'input', 'input_name': uncertainty_input_name} if uncertainty_input_name else {'kind': 'optimize'})
        ),
    }

    metadata = {
        'system': system_name,
        'grid_resolution': grid_resolution,
        'state_bounds': (bounds_lo.tolist(), bounds_hi.tolist()),
        'time_horizon': time_horizon,
        'time_steps': time_steps,
        'accuracy': accuracy,
        'periodic_dims': periodic_dims,
        # Persist coordinate vectors as numpy arrays (converted to torch on load)
        'grid_coordinate_vectors': [
            np.linspace(bounds_lo[i], bounds_hi[i], grid_resolution[i]).astype(np.float32)
            for i in range(len(grid_resolution))
        ],
        'dynamics_name': dynamics_name,
        'bindings': bindings,
    }
    
    # Convert to ascending time for persistence and usage:
    # Solver produced times ~ [H..0] (decreasing). Save as [0..H] (ascending).
    times_np = np.array(times)
    values_np = np.array(values)
    gradients_np = np.array(gradients)

    times_save = times_np[::-1].copy().astype(np.float32)
    values_save = values_np[::-1, ...].copy().astype(np.float32)
    gradients_save = gradients_np[::-1, ...].copy().astype(np.float32)

    value_function = GridValue(
        values=values_save,
        times=times_save,
        gradients=gradients_save,
        metadata=metadata,
        interpolate=False,
        hj_dynamics=dynamics,
    )
    
    # Write tagged payload (numpy arrays for portability)
    out_dir = Path('.cache') / 'grid_values'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.pkl"
    payload = {
        'tag': tag,
        'description': description,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dynamics_name': dynamics_name,
        'system_name': system_name,
        'values': values_save,
        'times': times_save,
        'gradients': gradients_save,
        'metadata': metadata,
    }
    tmp = out_path.with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
        f.flush()
    tmp.replace(out_path)
    print(f"✓ Saved GridValue cache: {out_path}")
    
    # Save lightweight metadata separately for fast listing
    meta_path = out_path.with_suffix('.meta.json')
    meta_export = {
        'tag': tag,
        'description': description,
        'created_at': payload['created_at'],
        'dynamics_name': dynamics_name,
        'system_name': system_name,
        'grid_shape': grid_resolution,
        'time_steps': time_steps,
        'accuracy': accuracy,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta_export, f, indent=2)
    
    # Summary
    print("\n✓ GridValue Build Complete!")
    
    # Lightweight summary (no dependency on GridValue.get_summary)
    try:
        grid_shape = value_function.grid_shape
        t0 = float(value_function._times[0])
        t1 = float(value_function._times[-1])
        vals = np.asarray(value_function._values)
        vmin = float(vals.min())
        vmax = float(vals.max())
        init_slice = vals[..., 0]
        final_slice = vals[..., -1]
        init_unsafe = float((init_slice < 0).mean())
        final_unsafe = float((final_slice < 0).mean())
        print(f"\nSummary:")
        print(f"  Grid shape: {grid_shape}")
        print(f"  Time range: [{t0:.3f}, {t1:.3f}]")
        print(f"  Value range: [{vmin:.3f}, {vmax:.3f}]")
        print(f"  Initial unsafe fraction: {init_unsafe:.2%}")
        print(f"  Final unsafe fraction: {final_unsafe:.2%}")
    except Exception as e:
        print(f"(Summary unavailable: {e})")
    
    return value_function


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config-driven usage:
  System presets come from config/systems.yaml.
  Resolution parameters come from config/resolutions.yaml.
  Use --set KEY=VALUE for ad-hoc overrides.

Examples:
  python build_grid_value.py --dynamics RoverDark --control-grid-set-tag RoverDark_MPC_Box --tag RoverDark_WorstCase
  python build_grid_value.py --dynamics RoverDark --preset recursive_brt --control-grid-set-tag RoverDark_MPC_Box --tag RoverDark_Recursive
  python build_grid_value.py --dynamics RoverDark --control-grid-set-tag RoverDark_MPC_Box --tag my_value --set accuracy=high
""",
    )

    # Essential args
    parser.add_argument('--list', action='store_true', help='List available dynamics and cache tags')
    parser.add_argument('--dynamics', type=str, help='Dynamics class name (e.g., RoverDark)')
    parser.add_argument('--tag', type=str, help='Output cache tag (filename without extension)')
    parser.add_argument('--config', type=str, default='config/resolutions.yaml', help='Path to grid resolution config')
    parser.add_argument('--force', action='store_true', help='Overwrite tag if exists')
    
    # Generic override mechanism
    parser.add_argument('--set', type=str, action='append', dest='overrides', metavar='KEY=VALUE',
                        help='Override config value (can be repeated)')

    # System configuration
    parser.add_argument('--system', type=str, help='System class name (default: inferred from GridSet)')
    parser.add_argument('--preset', type=str, help='Load system_args from config preset (e.g., "recursive_brt")')
    parser.add_argument('--system-arg', type=str, action='append', dest='system_args', metavar='KEY=VALUE',
                        help='Set system attribute (can be repeated). Overrides preset values.')

    # Channel bindings (convenience args)
    parser.add_argument('--control-grid-set-tag', type=str, help='Bind control as Set via GridSet tag')
    parser.add_argument('--control-grid-input-tag', type=str, help='Bind control as Input via GridInput tag')
    parser.add_argument('--control-input', type=str, help='Bind control as Input (class name)')
    parser.add_argument('--disturbance-grid-input-tag', type=str, help='Bind disturbance as Input via GridInput tag')
    parser.add_argument('--disturbance-input', type=str, help='Bind disturbance as Input (class name)')
    parser.add_argument('--uncertainty-grid-input-tag', type=str, help='Bind uncertainty as Input via GridInput tag')
    parser.add_argument('--uncertainty-input', type=str, help='Bind uncertainty as Input (class name)')

    # Other convenience args
    parser.add_argument('--description', type=str, default='', help='Optional description')
    parser.add_argument('--grid-resolution', type=int, nargs='+', help='Grid resolution (e.g., 50 50 25)')
    parser.add_argument('--state-bounds', type=float, nargs='+', help='State bounds [lo_1 ... lo_n hi_1 ... hi_n]')
    parser.add_argument('--time-horizon', type=float, help='Custom time horizon (seconds)')
    parser.add_argument('--time-steps', type=int, help='Number of time steps')
    parser.add_argument('--accuracy', type=str, default='very_high', choices=['low', 'medium', 'high', 'very_high'], help='Solver accuracy')
    parser.add_argument('--no-progress', action='store_true', help='Disable solver progress bar')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_available()
        return

    # Validate minimal args
    if not args.dynamics or not args.tag:
        parser.error("--dynamics and --tag are required (or use --list)")

    # Check destination
    out_dir = Path('.cache') / 'grid_values'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.pkl"
    if out_path.exists() and not args.force:
        print(f"\n⚠ GridValue tag already exists: {out_path}")
        print("Use --force to overwrite")
        return

    # Load preset system_args from systems.yaml if specified
    system_args_dict = {}
    if args.preset:
        # Need to know system_name to load preset
        # Try to resolve it from GridSet or GridInput metadata first
        preset_system = args.system
        if preset_system is None and args.control_grid_set_tag:
            gs_meta = get_grid_set_metadata(args.control_grid_set_tag)
            preset_system = gs_meta.get('system_name')
        if preset_system is None and args.control_grid_input_tag:
            gi_meta = get_grid_input_metadata(args.control_grid_input_tag)
            preset_system = gi_meta.get('system_name')
        if preset_system is None:
            parser.error("--preset requires --system, --control-grid-set-tag, or --control-grid-input-tag to determine which system's presets to load")
        
        # Load systems.yaml and get preset
        systems_cfg_path = Path('config/systems.yaml')
        if not systems_cfg_path.exists():
            parser.error(f"Systems config not found: {systems_cfg_path}")
        with systems_cfg_path.open('r') as f:
            systems_cfg = yaml.safe_load(f) or {}
        sys_cfg = systems_cfg.get(preset_system, {})
        presets = sys_cfg.get('presets', {})
        if args.preset not in presets:
            available = list(presets.keys()) if presets else ['(none)']
            parser.error(f"Preset '{args.preset}' not found for system '{preset_system}'. Available: {available}")
        preset_cfg = presets[args.preset]
        system_args_dict = dict(preset_cfg.get('system_args', {}))
        print(f"Loaded preset '{args.preset}': {system_args_dict}")

    # Parse CLI system args and merge (CLI overrides preset)
    if args.system_args:
        for arg in args.system_args:
            if '=' not in arg:
                parser.error(f"Invalid --system-arg format: '{arg}'. Expected KEY=VALUE")
            key, value = arg.split('=', 1)
            # Try to parse value as number or bool
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
            system_args_dict[key] = value

    # Build grid value
    build_value_function(
        args.dynamics,
        system_name=args.system,
        system_args=system_args_dict or None,
        control_grid_set_tag=args.control_grid_set_tag,
        control_grid_input_tag=args.control_grid_input_tag,
        control_input_name=args.control_input,
        disturbance_grid_input_tag=args.disturbance_grid_input_tag,
        disturbance_input_name=args.disturbance_input,
        uncertainty_grid_input_tag=args.uncertainty_grid_input_tag,
        uncertainty_input_name=args.uncertainty_input,
        tag=args.tag,
        description=args.description,
        grid_resolution=tuple(args.grid_resolution) if args.grid_resolution else None,
        state_bounds=tuple(args.state_bounds) if args.state_bounds else None,
        time_horizon=args.time_horizon,
        time_steps=args.time_steps,
        accuracy=args.accuracy,
        config_path=args.config,
        progress_bar=not args.no_progress,
    )


if __name__ == '__main__':
    main()
