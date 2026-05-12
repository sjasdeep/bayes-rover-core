"""Simulate a closed-loop system under specified inputs.

Outputs are saved under outputs/simulations/{TAG}/: results.pkl, metadata.json.
Use visualize_simulation.py to generate videos/frames from saved results.
Use inspect_simulation.py to view saved simulation data.
Supports batch mode via initial-state presets loaded from config/simulations.yaml.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils.registry import (
    get_available_input_classes,
    get_available_system_classes,
    instantiate_system,
)
from src.utils.cache_loaders import (
    get_grid_value_metadata,
    get_grid_set_metadata,
    get_grid_input_metadata,
    get_nn_input_metadata,
    resolve_input_with_class_and_tag,
    CACHE_BACKED_INPUT_TYPES,
)
from src.core.systems import System
from src.core.simulators import simulate_euler, simulate_discrete


def _list_caches_for_system(system_name: str, cache_type: str) -> List[str]:
    """List cache tags compatible with the given system.
    
    Args:
        system_name: Name of the system class
        cache_type: One of 'grid_values', 'grid_sets', 'grid_inputs', 'nn_inputs'
    
    Returns:
        List of compatible cache tags
    """
    cache_dir = Path('.cache') / cache_type
    if not cache_dir.exists():
        return []
    
    if cache_type == 'nn_inputs':
        files = sorted(cache_dir.glob('*.meta.json'))
    else:
        files = sorted(cache_dir.glob('*.pkl'))
    compatible = []
    
    for f in files:
        if cache_type == 'nn_inputs':
            # file name is {tag}.meta.json
            name = f.name
            tag = name[:-10] if name.endswith('.meta.json') else f.stem
        else:
            tag = f.stem
        try:
            if cache_type == 'grid_values':
                meta = get_grid_value_metadata(tag)
            elif cache_type == 'grid_sets':
                meta = get_grid_set_metadata(tag)
            elif cache_type == 'grid_inputs':
                meta = get_grid_input_metadata(tag)
            elif cache_type == 'nn_inputs':
                meta = get_nn_input_metadata(tag)
            else:
                continue
            
            if meta.get('system_name') == system_name:
                compatible.append(tag)
        except Exception:
            continue
    
    return compatible


def list_available(system_filter: Optional[str] = None) -> None:
    systems = get_available_system_classes()
    inputs = [c for c in get_available_input_classes() if c.__name__ != 'GridInput']
    
    # Filter to specific system if provided
    if system_filter:
        systems = [s for s in systems if s.__name__ == system_filter]
        if not systems:
            print(f"\nSystem '{system_filter}' not found.\n")
            return
    
    print("\nAvailable systems and inputs:\n")
    for s in sorted(systems, key=lambda c: c.__name__):
        compat = [i for i in inputs if hasattr(i, 'system_class') and issubclass(s, i.system_class)]
        ctrls = [i for i in compat if getattr(i, 'type', 'any') in ('any', 'control')]
        dists = [i for i in compat if getattr(i, 'type', 'any') in ('any', 'disturbance')]
        uncs = [i for i in compat if getattr(i, 'type', 'any') in ('any', 'uncertainty')]
        print(f"- {s.__name__}")
        if ctrls:
            print("  Control:")
            for c in sorted(ctrls, key=lambda c: c.__name__):
                print(f"    - {c.__name__}")
                # Show available GridValue caches for OptimalInputFromValue
                if c.__name__ == 'OptimalInputFromValue':
                    gv_tags = _list_caches_for_system(s.__name__, 'grid_values')
                    if gv_tags:
                        print("      Available GridValue caches:")
                        for tag in gv_tags[:10]:  # Limit to 10
                            print(f"        - {tag}")
                        if len(gv_tags) > 10:
                            print(f"        ... and {len(gv_tags) - 10} more")
        if dists:
            print("  Disturbance:")
            for d in sorted(dists, key=lambda c: c.__name__):
                print(f"    - {d.__name__}")
                # Show available GridValue caches for OptimalInputFromValue
                if d.__name__ == 'OptimalInputFromValue':
                    gv_tags = _list_caches_for_system(s.__name__, 'grid_values')
                    if gv_tags:
                        print("      Available GridValue caches:")
                        for tag in gv_tags[:10]:  # Limit to 10
                            print(f"        - {tag}")
                        if len(gv_tags) > 10:
                            print(f"        ... and {len(gv_tags) - 10} more")
        if uncs:
            print("  Uncertainty:")
            for u in sorted(uncs, key=lambda c: c.__name__):
                print(f"    - {u.__name__}")
                # Show available GridValue caches for OptimalInputFromValue
                if u.__name__ == 'OptimalInputFromValue':
                    gv_tags = _list_caches_for_system(s.__name__, 'grid_values')
                    if gv_tags:
                        print("      Available GridValue caches:")
                        for tag in gv_tags[:10]:  # Limit to 10
                            print(f"        - {tag}")
                        if len(gv_tags) > 10:
                            print(f"        ... and {len(gv_tags) - 10} more")
        
        # Show available caches for this system
        gi_tags = _list_caches_for_system(s.__name__, 'grid_inputs')
        gs_tags = _list_caches_for_system(s.__name__, 'grid_sets')
        gv_tags = _list_caches_for_system(s.__name__, 'grid_values')
        nn_tags = _list_caches_for_system(s.__name__, 'nn_inputs')

        if gi_tags or gs_tags or gv_tags or nn_tags:
            print("  Available Caches:")
            if gi_tags:
                print(f"    GridInputs: {len(gi_tags)} cache(s)")
                for tag in gi_tags[:3]:
                    print(f"      - {tag}")
                if len(gi_tags) > 3:
                    print(f"      ... and {len(gi_tags) - 3} more")
            if gs_tags:
                print(f"    GridSets: {len(gs_tags)} cache(s)")
                for tag in gs_tags[:3]:
                    print(f"      - {tag}")
                if len(gs_tags) > 3:
                    print(f"      ... and {len(gs_tags) - 3} more")
            if gv_tags:
                print(f"    GridValues: {len(gv_tags)} cache(s)")
                for tag in gv_tags[:3]:
                    print(f"      - {tag}")
                if len(gv_tags) > 3:
                    print(f"      ... and {len(gv_tags) - 3} more")
            if nn_tags:
                print(f"    NNInputs: {len(nn_tags)} cache(s)")
                for tag in nn_tags[:3]:
                    print(f"      - {tag}")
                if len(nn_tags) > 3:
                    print(f"      ... and {len(nn_tags) - 3} more")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config-driven usage:
  All parameters can be set in config/simulations.yaml and selected via --preset.
  Use --set KEY=VALUE for ad-hoc overrides without editing the config file.

Examples:
  # Run with config preset
  python simulate.py --system RoverDark --preset failure_batch --tag my_sim

  # Override specific values
  python simulate.py --system RoverDark --tag my_sim --set dt=0.02 --set control=GridInput --set control_tag=RoverDark_MPC

  # Override initial state
  python simulate.py --system RoverDark --tag my_sim --set initial_state=1,2,0

Available config keys (see config/simulations.yaml for full documentation):
  control, disturbance, uncertainty  - Input class names
  control_tag, disturbance_tag, uncertainty_tag - Cache tags for cache-backed inputs
  dt, time_horizon - Timing parameters
  initial_state, initial_states - Single or batch initial conditions
  control_grid_value_tag, uncertainty_grid_value_tag - GridValue references for scheduling inputs
""",
    )

    # Essential args
    parser.add_argument('--list', action='store_true', help='List available systems and inputs')
    parser.add_argument('--system', type=str, help='System class name (e.g., RoverDark)')
    parser.add_argument('--preset', type=str, dest='initial_states_preset', help='Preset name from config (overlays onto defaults)')
    parser.add_argument('--tag', type=str, help='Output tag (creates outputs/simulations/{TAG}/)')
    parser.add_argument('--config', type=str, dest='sim_config', default='config/simulations.yaml', help='Config file path')
    
    # Generic override mechanism
    parser.add_argument('--set', type=str, action='append', dest='overrides', metavar='KEY=VALUE',
                        help='Override config value (can be repeated). Examples: --set dt=0.05 --set control=GridInput')

    # Keep a few commonly-used convenience args for backwards compatibility
    parser.add_argument('--control', type=str, help='Control input class (convenience for --set control=X)')
    parser.add_argument('--disturbance', type=str, help='Disturbance input class (convenience for --set disturbance=X)')
    parser.add_argument('--uncertainty', type=str, help='Uncertainty input class (convenience for --set uncertainty=X)')
    parser.add_argument('--control-tag', type=str, help='Control cache tag (convenience for --set control_tag=X)')
    parser.add_argument('--disturbance-tag', type=str, help='Disturbance cache tag (convenience for --set disturbance_tag=X)')
    parser.add_argument('--uncertainty-tag', type=str, help='Uncertainty cache tag (convenience for --set uncertainty_tag=X)')
    parser.add_argument('--dt', type=float, help='Simulation timestep (convenience for --set dt=X)')
    parser.add_argument('--time-horizon', type=float, help='Time horizon override (convenience for --set time_horizon=X)')
    parser.add_argument('--description', type=str, help='Description (convenience for --set description=X)')
    parser.add_argument('--initial-state', type=float, nargs='+', help='Initial state (convenience for --set initial_state=X,Y,Z)')
    
    # Legacy aliases (deprecated, use --preset or --set)
    parser.add_argument('--initial-states-preset', type=str, dest='initial_states_preset_legacy', help=argparse.SUPPRESS)
    parser.add_argument('--sim-config', type=str, dest='sim_config_legacy', help=argparse.SUPPRESS)
    
    # Random initial states
    parser.add_argument('--n-random-states', type=int, help='Number of random initial states to sample uniformly within state_limits')
    parser.add_argument('--reject-initial-failures', action='store_true', help='Reject initial states that start in failure regions')
    parser.add_argument('--reject-value-grid', type=str, default=None, 
                        help='Reject states in sublevel set of a GridValue (tag). States with V(x,t=0)<=0 are rejected.')
    parser.add_argument('--reject-value-threshold', type=float, default=0.0,
                        help='Threshold for --reject-value-grid rejection (default: 0.0, i.e., reject V<=0)')
    parser.add_argument('--fix-initial-slice', type=str, action='append', metavar='DIM=VALUE',
                        help='Fix a state dimension to a constant value (e.g., --fix-initial-slice 3=0 to set s=0). Can be repeated.')
    parser.add_argument('--random-seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()
    
    # Handle legacy aliases
    if args.initial_states_preset_legacy and not args.initial_states_preset:
        args.initial_states_preset = args.initial_states_preset_legacy
    if args.sim_config_legacy and args.sim_config == 'config/simulations.yaml':
        args.sim_config = args.sim_config_legacy
    
    return args


def _prepare_inputs(system: System, control_name: str, dist_name: str, unc_name: str,
                    control_tag: Optional[str], dist_tag: Optional[str],
                    unc_tag: Optional[str],
                    total_sim_horizon: float,
                    sim_config: Optional[dict] = None):
    """Prepare control, disturbance, and uncertainty inputs using consolidated resolution."""
    control = resolve_input_with_class_and_tag(
        system, input_class=control_name, tag=control_tag, role="control",
        sim_config=sim_config
    )
    disturbance = resolve_input_with_class_and_tag(
        system, input_class=dist_name, tag=dist_tag, role="disturbance",
        sim_config=sim_config
    )
    uncertainty = resolve_input_with_class_and_tag(
        system, input_class=unc_name, tag=unc_tag, role="uncertainty",
        sim_config=sim_config
    )
    return control, disturbance, uncertainty


def _generate_random_initial_states(system: System,
                                    n_samples: int,
                                    reject_failures: bool = False,
                                    reject_value_grid = None,
                                    reject_value_threshold: float = 0.0,
                                    fixed_slices: Optional[dict[int, float]] = None,
                                    seed: Optional[int] = None,
                                    max_attempts_factor: int = 10) -> tuple[torch.Tensor, dict]:
    """Generate random initial states uniformly sampled within system.state_limits.
    
    Args:
        system: System instance with state_limits
        n_samples: Number of initial states to generate
        reject_failures: If True, reject states where failure_function(state) <= 0
        reject_value_grid: If provided, a GridValue whose sublevel set is rejected.
                          States with V(x, t=0) <= threshold are rejected.
        reject_value_threshold: Threshold for value grid rejection (default: 0.0)
        fixed_slices: Dict mapping dimension index to fixed value (e.g., {3: 0.0} to fix s=0)
        seed: Random seed for reproducibility
        max_attempts_factor: Factor for max attempts when rejecting failures (max_attempts = n_samples * factor)
    
    Returns:
        (points, meta) where points is [n_samples, state_dim] and meta has sampling info
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    limits = torch.as_tensor(system.state_limits, dtype=torch.float32)
    lo = limits[0]  # First row is lower bounds
    hi = limits[1]  # Second row is upper bounds
    state_dim = lo.shape[0]
    
    # Apply fixed slices by modifying effective bounds
    if fixed_slices:
        lo = lo.clone()
        hi = hi.clone()
        for dim, val in fixed_slices.items():
            lo[dim] = val
            hi[dim] = val
    
    do_rejection = reject_failures or (reject_value_grid is not None)
    
    if do_rejection:
        # Sample with rejection (vectorized)
        max_attempts = n_samples * max_attempts_factor
        collected = []
        total_attempts = 0
        n_rejected_failure = 0
        n_rejected_value = 0
        
        while len(collected) < n_samples and total_attempts < max_attempts:
            # Sample a batch - oversample to account for rejections
            remaining = n_samples - len(collected)
            batch_size = min(remaining * 4, max_attempts - total_attempts)
            samples = torch.rand(batch_size, state_dim) * (hi - lo) + lo
            
            valid_mask = torch.ones(batch_size, dtype=torch.bool)
            
            # Check failure function if requested
            if reject_failures:
                fail_vals = system.failure_function(samples)  # [batch_size, 1] or [batch_size]
                fail_vals = fail_vals.squeeze(-1) if fail_vals.ndim > 1 else fail_vals
                failure_mask = fail_vals <= 0
                n_rejected_failure += failure_mask.sum().item()
                valid_mask = valid_mask & ~failure_mask
            
            # Check value grid if provided
            if reject_value_grid is not None:
                # Query value at t=0 for all samples
                # Handle dimension mismatch: GridValue may be 3D while system is 4D
                gv_state_dim = reject_value_grid.state_dim
                if samples.shape[-1] > gv_state_dim:
                    # Project to first gv_state_dim dimensions
                    samples_for_gv = samples[..., :gv_state_dim]
                else:
                    samples_for_gv = samples
                
                v_vals = reject_value_grid.value(samples_for_gv, 0.0, interpolate=True)
                v_vals = v_vals.squeeze(-1) if v_vals.ndim > 1 else v_vals
                value_reject_mask = v_vals <= reject_value_threshold
                n_rejected_value += (valid_mask & value_reject_mask).sum().item()
                valid_mask = valid_mask & ~value_reject_mask
            
            valid_samples = samples[valid_mask]
            
            if valid_samples.shape[0] > 0:
                collected.append(valid_samples)
            
            total_attempts += batch_size
        
        if collected:
            all_valid = torch.cat(collected, dim=0)
        else:
            all_valid = torch.empty(0, state_dim)
            
        if all_valid.shape[0] < n_samples:
            raise ValueError(f"Could not sample {n_samples} valid initial states after {total_attempts} attempts. "
                           f"Only found {all_valid.shape[0]} valid states. Consider reducing n_random_states "
                           f"or disabling rejection options.")
        
        pts = all_valid[:n_samples]
        n_rejected = total_attempts - all_valid.shape[0]
    else:
        pts = torch.rand(n_samples, state_dim) * (hi - lo) + lo
        n_rejected = 0
        n_rejected_failure = 0
        n_rejected_value = 0
    
    meta = {
        'sampling_method': 'random_uniform',
        'n_samples': n_samples,
        'reject_failures': reject_failures,
        'reject_value_grid_tag': getattr(reject_value_grid, '_tag', None) if reject_value_grid else None,
        'reject_value_threshold': reject_value_threshold if reject_value_grid else None,
        'fixed_slices': fixed_slices,
        'n_rejected': n_rejected if do_rejection else None,
        'n_rejected_failure': n_rejected_failure if reject_failures else None,
        'n_rejected_value': n_rejected_value if reject_value_grid else None,
        'seed': seed,
        'state_limits': {
            'lo': lo.tolist(),
            'hi': hi.tolist(),
        },
    }
    return pts, meta


def _gather_initial_states(system: System, args: argparse.Namespace, cfg: dict) -> tuple[torch.Tensor, Optional[dict]]:
    # Priority:
    #   0) Random states: --n-random-states
    #   1) Explicit initial-state(s): CLI > config
    #   2) Fallback to system.initial_state
    
    # 0) Random sampling
    if getattr(args, 'n_random_states', None) is not None:
        n_random = args.n_random_states
        reject_failures = getattr(args, 'reject_initial_failures', False)
        reject_value_grid_tag = getattr(args, 'reject_value_grid', None)
        reject_value_threshold = getattr(args, 'reject_value_threshold', 0.0)
        seed = getattr(args, 'random_seed', None)
        
        # Parse fixed slices (e.g., --fix-initial-slice 3=0)
        fixed_slices = None
        fix_slice_args = getattr(args, 'fix_initial_slice', None)
        if fix_slice_args:
            fixed_slices = {}
            for spec in fix_slice_args:
                dim_str, val_str = spec.split('=')
                fixed_slices[int(dim_str)] = float(val_str)
        
        # Load value grid if specified
        reject_value_grid = None
        if reject_value_grid_tag:
            from src.utils.cache_loaders import load_grid_value_by_tag
            reject_value_grid = load_grid_value_by_tag(reject_value_grid_tag, interpolate=True)
            reject_value_grid._tag = reject_value_grid_tag  # Store tag for metadata
            print(f"Loaded GridValue '{reject_value_grid_tag}' for rejection (threshold={reject_value_threshold})")
        
        pts, meta = _generate_random_initial_states(
            system, n_random, reject_failures, 
            reject_value_grid=reject_value_grid,
            reject_value_threshold=reject_value_threshold,
            fixed_slices=fixed_slices,
            seed=seed
        )
        
        # Build descriptive source string
        src = f"CLI (--n-random-states {n_random})"
        if reject_failures and meta.get('n_rejected_failure'):
            n_rej = meta['n_rejected_failure']
            rate = n_rej / (n_random + n_rej) * 100
            src += f", rejected {n_rej} failure states ({rate:.1f}%)"
        if reject_value_grid and meta.get('n_rejected_value'):
            n_rej = meta['n_rejected_value']
            rate = n_rej / (n_random + n_rej) * 100
            src += f", rejected {n_rej} states in V<={reject_value_threshold} ({rate:.1f}%)"
        if fixed_slices:
            src += f", fixed slices: {fixed_slices}"
        if seed is not None:
            src += f", seed={seed}"
        print(f"Initial state(s): {n_random} random samples within state_limits (from {src})")
        return pts, meta

    # 1) Explicit single state via CLI
    states: List[List[float]] = []
    source = None
    if args.initial_state:
        states = [list(map(float, args.initial_state))]
        source = "CLI (--initial-state)"
    elif isinstance(cfg, dict):
        if 'initial_states' in cfg and isinstance(cfg['initial_states'], list):
            states = [list(map(float, s)) for s in cfg['initial_states']]
            source = "config (initial_states)"
        elif 'initial_state' in cfg and isinstance(cfg['initial_state'], list):
            if cfg['initial_state'] and isinstance(cfg['initial_state'][0], list):
                states = [list(map(float, s)) for s in cfg['initial_state']]
                source = "config (initial_state with multiple states)"
            else:
                states = [list(map(float, cfg['initial_state']))]
                source = "config (initial_state)"

    if not states:
        init = getattr(system, 'initial_state', None)
        if init is None:
            raise ValueError("System does not define an initial_state and none was provided")
        states = [list(map(float, init.tolist() if hasattr(init, 'tolist') else list(init)))]
        source = "system default (initial_state attribute)"

    print(f"Initial state(s): {len(states)} trajectory/trajectories (from {source})")

    t = torch.as_tensor(states, dtype=torch.float32)
    return t, None



def main() -> None:
    args = parse_args()

    # Propagate the simulations config path to downstream loaders that don't
    # receive it explicitly (e.g., Inputs calling load_simulation_config).
    try:
        from src.utils.config import set_default_simulations_config_path
        set_default_simulations_config_path(args.sim_config)
    except Exception:
        pass

    if args.list:
        list_available(system_filter=args.system)
        return

    # Validate required args when not listing
    required_min = ['system', 'tag']
    missing_min = [k for k in required_min if getattr(args, k) in (None, '')]
    if missing_min:
        raise SystemExit(f"Missing required args: {', '.join('--' + m for m in missing_min)} (or use --list)")

    # Load config for defaults (and optional preset overlay)
    from src.utils.config import load_simulation_config, parse_key_value_overrides, apply_overrides
    cfg = load_simulation_config(
        system_name=args.system,
        control_name=args.control,
        preset_name=args.initial_states_preset,
        path=args.sim_config,
    )
    
    # Apply CLI convenience args as overrides (before --set, so --set takes priority)
    cli_overrides = {}
    if args.control:
        cli_overrides['control'] = args.control
    if args.disturbance:
        cli_overrides['disturbance'] = args.disturbance
    if args.uncertainty:
        cli_overrides['uncertainty'] = args.uncertainty
    if args.control_tag:
        cli_overrides['control_tag'] = args.control_tag
    if args.disturbance_tag:
        cli_overrides['disturbance_tag'] = args.disturbance_tag
    if args.uncertainty_tag:
        cli_overrides['uncertainty_tag'] = args.uncertainty_tag
    if args.dt is not None:
        cli_overrides['dt'] = args.dt
    if args.time_horizon is not None:
        cli_overrides['time_horizon'] = args.time_horizon
    if args.description:
        cli_overrides['description'] = args.description
    if args.initial_state:
        cli_overrides['initial_state'] = args.initial_state

    if cli_overrides:
        cfg = apply_overrides(cfg, cli_overrides)
    
    # Apply --set overrides (highest priority)
    if args.overrides:
        set_overrides = parse_key_value_overrides(args.overrides)
        cfg = apply_overrides(cfg, set_overrides)
        print(f"Applied --set overrides: {set_overrides}")

    # Resolve system class and instantiate
    try:
        system: System = instantiate_system(args.system)
    except ValueError as e:
        raise SystemExit(str(e))
    
    # Optional: override time horizon before anything uses it
    if cfg.get('time_horizon') is not None:
        try:
            system.time_horizon = float(cfg['time_horizon'])
        except Exception as e:
            raise SystemExit(f"Failed to set system.time_horizon: {e}")
    
    # Determine device from system settings
    use_gpu = getattr(system, '_use_gpu', False)
    cuda_available = torch.cuda.is_available()
    if use_gpu and cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Log device selection details
    print(f"Device: {device} | CUDA available={cuda_available} | system._use_gpu={use_gpu}")

    # Resolve classes and params from config (CLI args already merged)
    control_name = cfg.get('control')
    dist_name = cfg.get('disturbance')
    unc_name = cfg.get('uncertainty')
    
    # Log where values came from
    print(f"Control: {control_name} (from config)")
    print(f"Disturbance: {dist_name} (from config)")
    print(f"Uncertainty: {unc_name} (from config)")
    
    # Ensure required classes are present
    for key, val in [('control', control_name), ('disturbance', dist_name), ('uncertainty', unc_name)]:
        if not val:
            raise SystemExit(f"Missing '{key}' in config. Set via --set {key}=CLASS or in {args.sim_config}.")

    # Require dt to be specified; derive steps from system.time_horizon
    dt = cfg.get('dt')
    if dt is None:
        raise SystemExit(f"Missing 'dt' in config. Set via --set dt=VALUE or in {args.sim_config}.")
    dt = float(dt)
    print(f"Time step (dt): {dt}")

    # Derive steps from the system's time horizon (possibly overridden)
    H = float(getattr(system, 'time_horizon'))
    steps = int(round(H / dt))
    total_sim_horizon = H
    print(f"Time horizon (H): {H}")
    print(f"Simulation steps: {steps}  [H/dt ≈ {H/dt:.3f}]")

    # Prepare inputs - tags come from config
    def _resolve_tag_for(input_name: str, cfg_tag: Optional[str]) -> Optional[str]:
        if input_name in CACHE_BACKED_INPUT_TYPES:
            return cfg_tag
        return None

    control_tag_eff = _resolve_tag_for(control_name, cfg.get('control_tag'))
    dist_tag_eff = _resolve_tag_for(dist_name, cfg.get('disturbance_tag'))
    unc_tag_eff = _resolve_tag_for(unc_name, cfg.get('uncertainty_tag'))

    control, disturbance, uncertainty = _prepare_inputs(
        system=system,
        control_name=control_name,
        dist_name=dist_name,
        unc_name=unc_name,
        control_tag=control_tag_eff,
        dist_tag=dist_tag_eff,
        unc_tag=unc_tag_eff,
        total_sim_horizon=total_sim_horizon,
        sim_config=cfg,
    )
    
    # Log where tags came from (if applicable)
    if control_tag_eff:
        src = 'CLI' if args.control_tag else 'config'
        print(f"Control tag: {control_tag_eff} (from {src})")
    if dist_tag_eff:
        src = 'CLI' if args.disturbance_tag else 'config'
        print(f"Disturbance tag: {dist_tag_eff} (from {src})")
    if unc_tag_eff:
        src = 'CLI' if args.uncertainty_tag else 'config'
        print(f"Uncertainty tag: {unc_tag_eff} (from {src})")

    def _maybe_to_device(obj):
        if hasattr(obj, "to"):
            try:
                obj.to(device)
            except TypeError:
                obj.to(device=device)  # type: ignore[arg-type]
            except Exception:
                pass

    for obj in (control, disturbance, uncertainty):
        _maybe_to_device(obj)

    # Initial states (batch)
    initial_states, initial_states_grid_meta = _gather_initial_states(system, args, cfg)
    # Validate dimensionality early for clearer errors
    try:
        sd = int(getattr(system, 'state_dim', initial_states.shape[1]))
    except Exception:
        sd = initial_states.shape[1]
    if int(initial_states.shape[1]) != sd:
        src = args.initial_states_preset or 'provided states'
        raise SystemExit(
            f"Initial state dimension mismatch: got {int(initial_states.shape[1])}, expected {sd}. "
            f"If using --initial-states-preset, check '{args.sim_config}' under system '{args.system}', preset '{src}'."
        )
    initial_states = initial_states.to(device)
    initial_states_cpu = initial_states.detach().cpu()
    
    # Pull batch size setting from system
    batch_size = getattr(system, '_batch_size', None)
    if batch_size is None:
        print(f"\nBatch size: Not set (system has no _batch_size attribute), will simulate all trajectories in one batch")
    else:
        print(f"\nBatch size: {batch_size} (from System._batch_size attribute)")
    
    print(f"\nSimulation configuration:")
    print(f"  Initial states: {initial_states.shape[0]}")
    print(f"  use_gpu={use_gpu}")
    if use_gpu and not cuda_available:
        print(f"  Device: {device} (CUDA not available, falling back to cpu)")
    else:
        print(f"  Device: {device}")

    # Simulate all trajectories in batches
    out_dir = Path('outputs') / 'simulations' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate all trajectories in batches
    out_dir = Path('outputs') / 'simulations' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Accumulate results from all batches
    all_states = []
    all_controls = []
    all_disturbances = []
    all_uncertainties = []
    all_estimated_states = []
    times = None
    
    # Determine batch size for simulation
    n_trajectories = initial_states.shape[0]

    # Choose simulation stepping function
    simulate_fn = simulate_discrete if callable(getattr(system, 'next_state', None)) else simulate_euler
    sim_mode = 'discrete (next_state)' if simulate_fn is simulate_discrete else 'continuous (Euler)'
    print(f"\nSimulator: {sim_mode}")
    if batch_size is None or batch_size >= n_trajectories:
        # Simulate all at once
        if batch_size is None:
            print(f"\nNo batch size limit set, simulating all {n_trajectories} trajectories in one batch")
        else:
            print(f"\nBatch size ({batch_size}) >= n_trajectories ({n_trajectories}), simulating all in one batch")
        batched_result = simulate_fn(
            system=system,
            control=control,
            disturbance=disturbance,
            uncertainty=uncertainty,
            dt=dt,
            num_steps=steps,
            initial_state=initial_states,
            show_progress=True,
            leave_progress=True,
            device=device,
        )
        
        # Store batched results directly
        all_states.append(batched_result.states)
        all_controls.append(batched_result.controls)
        all_disturbances.append(batched_result.disturbances)
        all_uncertainties.append(batched_result.uncertainties)
        all_estimated_states.append(batched_result.estimated_states)
        times = batched_result.times
    else:
        # Simulate in batches
        print(f"\nSimulating {n_trajectories} trajectories in batches of {batch_size}...")
        for batch_start in range(0, n_trajectories, batch_size):
            batch_end = min(batch_start + batch_size, n_trajectories)
            batch_states = initial_states[batch_start:batch_end]
            
            print(f"  Batch {batch_start//batch_size + 1}/{(n_trajectories + batch_size - 1)//batch_size}: "
                  f"trajectories {batch_start} to {batch_end-1}")
            
            batched_result = simulate_fn(
                system=system,
                control=control,
                disturbance=disturbance,
                uncertainty=uncertainty,
                dt=dt,
                num_steps=steps,
                initial_state=batch_states,
                show_progress=True,
                leave_progress=True,
                device=device,
            )
            
            # Accumulate batch results
            all_states.append(batched_result.states)
            all_controls.append(batched_result.controls)
            all_disturbances.append(batched_result.disturbances)
            all_uncertainties.append(batched_result.uncertainties)
            all_estimated_states.append(batched_result.estimated_states)
            times = batched_result.times  # Same for all batches
    
    # Concatenate all batches into single tensors
    states = torch.cat(all_states, dim=0)  # [n_trajectories, time_steps+1, state_dim]
    controls = torch.cat(all_controls, dim=0)  # [n_trajectories, time_steps, control_dim]
    disturbances = torch.cat(all_disturbances, dim=0)  # [n_trajectories, time_steps, disturbance_dim]
    uncertainties = torch.cat(all_uncertainties, dim=0)  # [n_trajectories, time_steps, uncertainty_dim]
    estimated_states = torch.cat(all_estimated_states, dim=0)  # [n_trajectories, time_steps, state_dim]
    
    # Move to CPU for saving
    def _to_cpu_tensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu()
        return tensor

    states = _to_cpu_tensor(states)
    controls = _to_cpu_tensor(controls)
    disturbances = _to_cpu_tensor(disturbances)
    uncertainties = _to_cpu_tensor(uncertainties)
    estimated_states = _to_cpu_tensor(estimated_states)
    times = _to_cpu_tensor(times)

    # Save results as pickle with tensors and metadata (following cache pattern)
    results_path = out_dir / 'results.pkl'
    
    # Determine description source
    description = args.description if args.description is not None else cfg.get('description', '')
    if args.description is not None:
        print(f"Description: '{description}' (from CLI)")
    elif cfg.get('description'):
        print(f"Description: '{description}' (from config)")
    else:
        print("Description: (none)")
    
    # Build comprehensive payload with metadata
    payload = {
        'tag': args.tag,
        'description': description,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'system_name': args.system,
        'control_name': control_name,
        'disturbance_name': dist_name,
        'uncertainty_name': unc_name,
        'control_tag': (args.control_tag or cfg.get('control_tag')),
        'disturbance_tag': (args.disturbance_tag or cfg.get('disturbance_tag')),
        'uncertainty_tag': (args.uncertainty_tag or cfg.get('uncertainty_tag')),
        'dt': float(dt),
        'steps': int(steps),
        'time_horizon': float(total_sim_horizon),
        'initial_states': initial_states_cpu,  # Keep as tensor [n_trajectories, state_dim]
        'initial_states_grid_meta': initial_states_grid_meta,  # None unless grid slice used
        'device': str(device),
        'use_gpu': use_gpu,
        'batch_size': batch_size,
        'n_trajectories': n_trajectories,
        # Store batched results as tensors (all trajectories together)
        'states': states,  # [n_trajectories, time_steps+1, state_dim]
        'controls': controls,  # [n_trajectories, time_steps, control_dim]
        'disturbances': disturbances,  # [n_trajectories, time_steps, disturbance_dim]
        'uncertainties': uncertainties,  # [n_trajectories, time_steps, uncertainty_dim]
        'estimated_states': estimated_states,  # [n_trajectories, time_steps, state_dim]
        'times': times,  # [time_steps+1]
    }
    
    # Write atomically using temporary file
    tmp_path = results_path.with_suffix('.pkl.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
    tmp_path.replace(results_path)
    print(f"✓ Saved simulation results: {results_path}")
    
    # Save lightweight metadata separately for fast listing
    meta_path = out_dir / 'metadata.json'
    meta_export = {
        'tag': args.tag,
        'description': payload['description'],
        'created_at': payload['created_at'],
        'system_name': args.system,
        'control_name': control_name,
        'disturbance_name': dist_name,
        'uncertainty_name': unc_name,
        'dt': float(dt),
        'steps': int(steps),
        'time_horizon': float(total_sim_horizon),
        'n_trajectories': n_trajectories,
        'device': str(device),
        'initial_states_grid_meta': initial_states_grid_meta,
    }
    with open(meta_path, 'w') as mf:
        json.dump(meta_export, mf, indent=2)

    # Save human-readable summary
    summary_path = out_dir / 'summary.txt'
    with open(summary_path, 'w') as sf:
        sf.write(f"Simulation Summary: {args.tag}\n")
        sf.write("=" * 50 + "\n\n")
        sf.write(f"System: {args.system}\n")
        sf.write(f"Control: {control_name}\n")
        sf.write(f"Disturbance: {dist_name}\n")
        sf.write(f"Uncertainty: {unc_name}\n\n")
        sf.write(f"Time horizon: {total_sim_horizon:.2f}s\n")
        sf.write(f"Time step (dt): {dt:.4f}s\n")
        sf.write(f"Simulation steps: {steps}\n")
        sf.write(f"Trajectories: {n_trajectories}\n\n")
        sf.write(f"Created: {payload['created_at']}\n")
        if payload['description']:
            sf.write(f"Description: {payload['description']}\n")

    print(f"\n✓ Simulation complete. Outputs written to: {out_dir}")
    print(f"  - Results: {results_path}")
    print(f"  - Metadata: {meta_path}")
    print(f"  - Summary: {summary_path}")
    print(f"\nTo visualize: python scripts/simulation/visualize_simulation.py --tag {args.tag}")
    print(f"To inspect: python scripts/simulation/inspect_simulation.py --tag {args.tag}")


if __name__ == '__main__':
    main()
