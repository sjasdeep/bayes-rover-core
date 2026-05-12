"""Utilities to load cached GridInput, GridSet, GridValue, and NNInput by tag.

These helpers keep cache concerns out of core configuration classes. They
reconstruct query-only objects from payloads under .cache/, bind them to a
System, and return ready-to-use instances.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from src.core.inputs import Input
from src.core.sets import Set
from src.core.systems import System
from src.impl.inputs.derived.grid_input import GridInput
from src.impl.inputs.derived.nn_input import NNInput
from src.impl.inputs.derived.optimal_input_from_value import OptimalInputFromValue
from src.impl.sets.grid_set import GridSet
from src.impl.values.grid_value import GridValue
from src.utils.registry import get_hj_dynamics_class, get_input_class, instantiate_input, instantiate_system

__all__ = [
    "load_grid_input_by_tag",
    "load_grid_set_by_tag",
    "load_grid_value_by_tag",
    "load_nn_input_by_tag",
    "load_input_by_tag",
    "get_grid_input_metadata",
    "get_grid_set_metadata",
    "get_grid_value_metadata",
    "get_nn_input_metadata",
    "get_grid_input_payload",
    "get_grid_set_payload",
    "get_grid_value_payload",
    "instantiate_system_by_name",
    "resolve_hj_dynamics_class",
    # Consolidated resolution helpers
    "resolve_input",
    "resolve_set",
    "resolve_input_with_class_and_tag",
    "CACHE_BACKED_INPUT_TYPES",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _cache_path(subdir: str, tag: str, ext: str = ".pkl") -> Path:
    return Path(".cache") / subdir / f"{tag}{ext}"


# Re-export for backwards compatibility (prefer registry.instantiate_system)
instantiate_system_by_name = instantiate_system
resolve_hj_dynamics_class = get_hj_dynamics_class


# ---------------------------------------------------------------------------
# GridInput
# ---------------------------------------------------------------------------

def load_grid_input_by_tag(tag: str, system: System, *, interpolate: bool = True) -> GridInput:
    data = _load_pickle(_cache_path("grid_inputs", tag))
    gi = GridInput(
        wrapped_input=None,
        grid_cache=data["grid_cache"],
        state_grid_points=data["state_grid_points"],
        time_grid_points=data["time_grid_points"],
        interpolate=interpolate,
    )
    gi.bind(system)
    return gi


def get_grid_input_metadata(tag: str) -> Dict[str, Any]:
    data = _load_pickle(_cache_path("grid_inputs", tag))
    return {
        "system_name": data["system_name"],
        "input_name": data["input_name"],
        "input_type": data["input_type"],
        "grid_shape": list(data["grid_cache"].shape),
        "description": data["description"],
        "created_at": data["created_at"],
    }


def get_grid_input_payload(tag: str) -> Dict[str, Any]:
    return _load_pickle(_cache_path("grid_inputs", tag))


# ---------------------------------------------------------------------------
# GridSet
# ---------------------------------------------------------------------------

def load_grid_set_by_tag(tag: str, system: System, *, interpolate: bool = True) -> GridSet:
    data = _load_pickle(_cache_path("grid_sets", tag))
    gs = GridSet(
        set_type=data["set_type"],
        state_grid_points=data["state_grid_points"],
        time_grid_points=data["time_grid_points"],
        box_lower=data.get("box_lower"),
        box_upper=data.get("box_upper"),
        hull_vertices=data.get("hull_vertices"),
        hull_vertices_padded=data.get("hull_vertices_padded"),
        hull_vertices_mask=data.get("hull_vertices_mask"),
        hull_state_idx_padded=data.get("hull_state_idx_padded"),
        box_state_est_corner_idx=data.get("box_state_est_corner_idx"),
        interpolate=interpolate,
    )
    gs.bind(system)
    return gs


def get_grid_set_metadata(tag: str) -> Dict[str, Any]:
    data = _load_pickle(_cache_path("grid_sets", tag))
    return {
        "system_name": data["system_name"],
        "input_name": data["input_name"],
        "set_type": data["set_type"],
        "grid_input_tag": data["grid_input_tag"],
        "nn_input_tag": data.get("nn_input_tag"),
        "grid_shape": data["grid_shape"],
        "description": data["description"],
        "created_at": data["created_at"],
    }


def get_grid_set_payload(tag: str) -> Dict[str, Any]:
    return _load_pickle(_cache_path("grid_sets", tag))


# ---------------------------------------------------------------------------
# GridValue
# ---------------------------------------------------------------------------

def _bind_channel(hj_dyn, bind_method, binding: Dict, system: System) -> None:
    """Bind a single channel (control/disturbance/uncertainty) from cached binding info."""
    kind = binding.get("kind")
    if kind == "set":
        bind_method(load_grid_set_by_tag(binding["grid_set_tag"], system))
    elif kind == "input":
        grid_tag = binding.get("grid_input_tag")
        if grid_tag:
            bind_method(load_grid_input_by_tag(grid_tag, system))
        else:
            bind_method(instantiate_input(binding["input_name"], system))


def load_grid_value_by_tag(tag: str, *, interpolate: bool = False) -> GridValue:
    """Load a cached GridValue and reconstruct its dynamics bindings."""
    data = _load_pickle(_cache_path("grid_values", tag))
    meta = data["metadata"]
    system = instantiate_system(data["system_name"])
    
    # Get runtime wrapper via the dynamics class interface
    # (runtime class uses PyTorch GridSet directly, not JaxGridSet)
    hj_dynamics_cls = get_hj_dynamics_class(data["dynamics_name"])
    if hj_dynamics_cls is None:
        raise ValueError(f"Unknown dynamics class: {data['dynamics_name']}")
    wrapper_cls = hj_dynamics_cls.runtime_class()
    hj_dyn = wrapper_cls(system)
    hj_dyn.system_instance = system

    bindings = meta["bindings"]
    ctrl_kind = bindings["control"].get("kind")
    ctrl_bind = hj_dyn.bind_control_set if ctrl_kind == "set" else hj_dyn.bind_control_input
    _bind_channel(hj_dyn, ctrl_bind, bindings["control"], system)
    _bind_channel(hj_dyn, hj_dyn.bind_disturbance_input, bindings["disturbance"], system)
    _bind_channel(hj_dyn, hj_dyn.bind_uncertainty_input, bindings["uncertainty"], system)
    hj_dyn.validate()

    return GridValue(
        values=data["values"],
        times=data["times"],
        gradients=data["gradients"],
        metadata=meta,
        interpolate=interpolate,
        hj_dynamics=hj_dyn,
    )


def get_grid_value_metadata(tag: str) -> Dict[str, Any]:
    """Load GridValue metadata (fast path via .meta.json if available)."""
    meta_path = _cache_path("grid_values", tag, ".meta.json")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    # Fallback: load from full pickle
    data = _load_pickle(_cache_path("grid_values", tag))
    meta = data["metadata"]
    return {
        "tag": tag,
        "description": data["description"],
        "created_at": data["created_at"],
        "dynamics_name": data["dynamics_name"],
        "system_name": data["system_name"],
        "grid_shape": tuple(len(cv) for cv in meta["grid_coordinate_vectors"]),
        "time_steps": meta["time_steps"],
        "accuracy": meta["accuracy"],
    }


def get_grid_value_payload(tag: str) -> Dict[str, Any]:
    return _load_pickle(_cache_path("grid_values", tag))


# ---------------------------------------------------------------------------
# NNInput
# ---------------------------------------------------------------------------

def load_nn_input_by_tag(tag: str, system: System, *, device: Optional[torch.device] = None) -> NNInput:
    """Load an NNInput from .cache/nn_inputs/{tag}.pth + .meta.json."""
    base = Path(".cache") / "nn_inputs" / tag
    nn = NNInput(str(base), device=str(device) if device else None)
    nn.bind(system)
    return nn


def get_nn_input_metadata(tag: str) -> Dict[str, Any]:
    """Read NNInput metadata from .cache/nn_inputs/{tag}.meta.json."""
    with open(_cache_path("nn_inputs", tag, ".meta.json"), "r") as f:
        meta = json.load(f)
    return {
        "system_name": meta.get("system_name"),
        "input_name": meta.get("input_name"),
        "input_type": meta.get("input_type"),
        "sizes": meta.get("sizes"),
        "time_invariant": meta.get("time_invariant", True),
        "description": meta.get("description", ""),
        "created_at": meta.get("created_at", ""),
        "training_in_progress": meta.get("training_in_progress", False),
        "checkpoint": meta.get("checkpoint"),
    }


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_input_by_tag(
    input_class: str,
    tag: str,
    system: System,
    *,
    role: str,
    device: Optional[torch.device] = None,
    interpolate: bool = False,
) -> Input:
    """Load a cache-backed Input by class name and tag.

    Args:
        input_class: One of 'GridInput', 'OptimalInputFromValue', 'NNInput'
        tag: Cache tag
        system: Target system to bind inputs to
        role: Channel role: 'control' | 'disturbance' | 'uncertainty'
        device: Optional device (for NNInput)
        interpolate: Enable interpolation where applicable
    """
    loaders = {
        "GridInput": ("grid_inputs", ".pkl", lambda: load_grid_input_by_tag(tag, system, interpolate=interpolate)),
        "OptimalInputFromValue": ("grid_values", ".pkl", lambda: OptimalInputFromValue(load_grid_value_by_tag(tag, interpolate=interpolate), channel=role)),
        "NNInput": ("nn_inputs", ".meta.json", lambda: load_nn_input_by_tag(tag, system, device=device)),
    }
    if input_class not in loaders:
        raise ValueError(f"Unsupported input_class '{input_class}'. Expected one of {set(loaders.keys())}.")
    
    subdir, ext, loader = loaders[input_class]
    path = _cache_path(subdir, tag, ext)
    if not path.exists():
        raise ValueError(f"Tag '{tag}' not found in '.cache/{subdir}' for {role} (class={input_class}).")
    return loader()


# ---------------------------------------------------------------------------
# Consolidated Input/Set Resolution
# ---------------------------------------------------------------------------

# Cache-backed input types that require a tag
CACHE_BACKED_INPUT_TYPES = frozenset({"GridInput", "NNInput", "OptimalInputFromValue"})


def resolve_input(
    system: System,
    *,
    input_class: Optional[str] = None,
    grid_input_tag: Optional[str] = None,
    nn_input_tag: Optional[str] = None,
    optimal_value_tag: Optional[str] = None,
    role: str = "control",
    interpolate: bool = True,
    device: Optional[str] = None,
) -> Tuple[Optional[Input], Optional[str]]:
    """Resolve an Input from class name or cache tag.
    
    Mutually exclusive options (specify at most one):
      - input_class: Raw Input class name (e.g., 'RoverDark_MPC')
      - grid_input_tag: Load GridInput from cache
      - nn_input_tag: Load NNInput from cache
      - optimal_value_tag: Load OptimalInputFromValue from cache
    
    Args:
        system: System to bind the input to
        input_class: Raw Input class name (instantiated directly)
        grid_input_tag: Cache tag for GridInput
        nn_input_tag: Cache tag for NNInput  
        optimal_value_tag: Cache tag for OptimalInputFromValue
        role: Channel role ('control', 'disturbance', 'uncertainty')
        interpolate: Enable interpolation for grid-based inputs
        device: Device for NNInput (e.g., 'cuda', 'cpu')
    
    Returns:
        (input_instance, resolved_name) or (None, None) if nothing specified
        
    Raises:
        ValueError: If multiple options specified or resolution fails
    """
    opts = [
        ("input_class", input_class),
        ("grid_input_tag", grid_input_tag),
        ("nn_input_tag", nn_input_tag),
        ("optimal_value_tag", optimal_value_tag),
    ]
    specified = [(name, val) for name, val in opts if val is not None]
    
    if len(specified) > 1:
        names = [name for name, _ in specified]
        raise ValueError(f"Specify only one of: {', '.join(names)}")
    
    if not specified:
        return None, None
    
    opt_name, opt_val = specified[0]
    
    if opt_name == "input_class":
        # Raw Input class - instantiate directly
        inp = instantiate_input(opt_val, system)
        return inp, opt_val
    elif opt_name == "grid_input_tag":
        inp = load_grid_input_by_tag(opt_val, system, interpolate=interpolate)
        return inp, f"GridInput:{opt_val}"
    elif opt_name == "nn_input_tag":
        dev = torch.device(device) if device else None
        inp = load_nn_input_by_tag(opt_val, system, device=dev)
        return inp, f"NNInput:{opt_val}"
    elif opt_name == "optimal_value_tag":
        gv = load_grid_value_by_tag(opt_val, interpolate=interpolate)
        inp = OptimalInputFromValue(gv, channel=role)
        return inp, f"OptimalInputFromValue:{opt_val}"
    
    return None, None


def resolve_set(
    system: System,
    *,
    grid_set_tag: Optional[str] = None,
    interpolate: bool = True,
) -> Tuple[Optional[Set], Optional[str]]:
    """Resolve a Set from cache tag.
    
    Args:
        system: System to bind the set to
        grid_set_tag: Cache tag for GridSet
        interpolate: Enable interpolation
    
    Returns:
        (set_instance, resolved_name) or (None, None) if nothing specified
    """
    if grid_set_tag is None:
        return None, None
    
    gs = load_grid_set_by_tag(grid_set_tag, system, interpolate=interpolate)
    return gs, f"GridSet:{grid_set_tag}"


def resolve_input_with_class_and_tag(
    system: System,
    *,
    input_class: Optional[str] = None,
    tag: Optional[str] = None,
    role: str = "control",
    interpolate: bool = True,
    device: Optional[str] = None,
    sim_config: Optional[Dict[str, Any]] = None,
) -> Input:
    """Resolve an Input from class name and optional tag.
    
    This is the pattern used by simulate.py where:
      - input_class is the class name (e.g., 'RoverDark_MPC', 'GridInput', 'NNInput')
      - tag is required only for cache-backed types (GridInput, NNInput, OptimalInputFromValue)
    
    Args:
        system: System to bind the input to
        input_class: Input class name (raw class or cache-backed type)
        tag: Cache tag (required for cache-backed types)
        role: Channel role ('control', 'disturbance', 'uncertainty')
        interpolate: Enable interpolation for grid-based inputs
        device: Device for NNInput
        sim_config: Merged simulation config dict; passed to input constructors
    
    Returns:
        Input instance (bound to system)
        
    Raises:
        ValueError: If cache-backed type specified without tag, or unknown class
    """
    if input_class is None:
        raise ValueError(f"Input class must be specified for {role}")
    
    # Build kwargs from sim_config for input constructor
    # Only pass keys that the input class actually accepts
    input_kwargs: Dict[str, Any] = {}
    if sim_config:
        # Get the input class to check its __init__ signature
        input_cls = get_input_class(input_class)
        if input_cls is not None:
            import inspect
            try:
                sig = inspect.signature(input_cls.__init__)
                accepted_params = set(sig.parameters.keys()) - {'self'}
                
                # Role-specific config key mapping:
                # - control inputs get control_grid_value_tag from config
                # - uncertainty inputs get uncertainty_grid_value_tag from config
                # Both are passed as grid_value_tag to the constructor.
                if role == 'uncertainty':
                    if 'grid_value_tag' in accepted_params and 'uncertainty_grid_value_tag' in sim_config:
                        input_kwargs['grid_value_tag'] = sim_config['uncertainty_grid_value_tag']
                else:
                    if 'grid_value_tag' in accepted_params and 'control_grid_value_tag' in sim_config:
                        input_kwargs['grid_value_tag'] = sim_config['control_grid_value_tag']
                
                # Pass any other accepted parameters that exist in sim_config
                for param in accepted_params:
                    if param in sim_config and param not in input_kwargs:
                        input_kwargs[param] = sim_config[param]
            except (ValueError, TypeError):
                pass  # Can't inspect, skip kwargs
    
    # Cache-backed types require a tag
    if input_class in CACHE_BACKED_INPUT_TYPES:
        if tag is None:
            raise ValueError(
                f"--{role}-tag is required when --{role} is one of {{{', '.join(sorted(CACHE_BACKED_INPUT_TYPES))}}}"
            )
        inp = load_input_by_tag(
            input_class=input_class,
            tag=tag,
            system=system,
            role=role,
            interpolate=interpolate,
            device=torch.device(device) if device else None,
        )
    else:
        # Raw Input class - tag should not be provided
        if tag is not None:
            raise ValueError(
                f"--{role}-tag was provided but --{role}={input_class} is not a cache-backed type. "
                f"Remove --{role}-tag or set --{role} to one of {{{', '.join(sorted(CACHE_BACKED_INPUT_TYPES))}}}."
            )
        # Don't bind yet - set_type must be called first for inputs like ZeroInput
        inp = instantiate_input(input_class, None, **input_kwargs)
    
    inp.set_type(role)
    inp.bind(system)
    return inp
