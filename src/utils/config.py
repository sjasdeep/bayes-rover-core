"""Centralized configuration utilities for resolution, visualization, and simulation presets.

This module consolidates reading config YAMLs so scripts don't duplicate parsing
and key selection logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

__all__ = [
    "set_default_simulations_config_path",
    "parse_key_value_overrides",
    "apply_overrides",
    "load_resolution_config",
    "load_visualization_presets",
    "load_simulation_presets",
    "load_simulation_config",
    "load_nn_training_config",
    "load_monte_carlo_config",
]

# Allow simulation scripts to set a default simulations.yaml path that
# other modules (e.g., Inputs) can inherit when they call loaders without
# an explicit path. This keeps a single source of truth per run.
_SIM_CONFIG_OVERRIDE_PATH: Optional[str] = None


def set_default_simulations_config_path(path: str) -> None:
    """Set a process-wide default path for simulations config.

    Modules that call load_simulation_config without specifying `path`
    will use this override if set.
    """
    global _SIM_CONFIG_OVERRIDE_PATH
    _SIM_CONFIG_OVERRIDE_PATH = str(path)


def parse_key_value_overrides(overrides: list[str] | None) -> Dict[str, Any]:
    """Parse KEY=VALUE strings into a dictionary.
    
    Supports:
      - Simple values: dt=0.05, tag=my_sim
      - Nested keys: control.tag=RoverDark_MPC
      - Lists: initial_state=[1,2,3] or initial_state=1,2,3
      - Booleans: verbose=true, force=false
      - Numbers: dt=0.05, steps=100
    
    Examples:
        parse_key_value_overrides(["dt=0.05", "control=GridInput"])
        # => {"dt": 0.05, "control": "GridInput"}
        
        parse_key_value_overrides(["initial_state=1,2,3"])
        # => {"initial_state": [1.0, 2.0, 3.0]}
    """
    if not overrides:
        return {}
    
    result: Dict[str, Any] = {}
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"Invalid override format: '{item}'. Expected KEY=VALUE.")
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Parse value type
        parsed = _parse_value(value)
        
        # Handle nested keys (e.g., "control.tag" -> {"control": {"tag": ...}})
        if '.' in key:
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = parsed
        else:
            result[key] = parsed
    
    return result


def _parse_value(value: str) -> Any:
    """Parse a string value into appropriate Python type."""
    # Boolean
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.lower() == 'none':
        return None
    
    # List (bracketed or comma-separated numbers)
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(v.strip()) for v in inner.split(',')]
    
    # Comma-separated values that look like numbers
    if ',' in value:
        parts = value.split(',')
        try:
            return [float(p.strip()) for p in parts]
        except ValueError:
            pass  # Fall through to string
    
    # Number (int or float)
    try:
        if '.' in value or 'e' in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # String (default)
    return value


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply overrides to a config dict (shallow merge at top level).
    
    Returns a new dict without mutating the original.
    """
    return _merge_shallow(config, overrides)


def load_resolution_config(system_name: str, input_name: Optional[str] = None, config_path: str = 'config/resolutions.yaml') -> Dict[str, Any]:
    """Load grid/time resolution configuration for the specified system/input.
    
    Falls back to 'default' entry if the specific input_name is not found.
    """
    cfg_path = Path(config_path)
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f) or {}
    sys_cfg = cfg[system_name]
    if input_name is None:
        return sys_cfg['default']
    # Try specific input name first, fall back to default
    if input_name in sys_cfg:
        return sys_cfg[input_name]
    return sys_cfg.get('default', {})


def load_visualization_presets(
    system_name: str,
    input_name: str = 'default',
    preset_name: Optional[str] = None,
    *,
    path: str = 'config/visualizations.yaml',
) -> Dict[str, Any]:
    """Load visualization presets from config/visualizations.yaml.

    Returns the dictionary for the given (system_name, input_name) subtree.
    Falls back to 'default' entry if the specific input_name is not found.
    If preset_name is provided, returns presets[preset_name] or {} if missing.
    """
    cfg_path = Path(path)
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f) or {}
    sys_node = cfg.get(system_name, {})
    # Try specific input name first, fall back to default
    if input_name in sys_node:
        in_node = sys_node[input_name]
    else:
        in_node = sys_node.get('default', {})
    
    if preset_name is None:
        return in_node
    return in_node.get(preset_name, {})


def load_simulation_presets(
    system_name: str,
    control_name: str = 'default',
    preset_name: Optional[str] = None,
    *,
    path: str = 'config/simulations.yaml',
) -> Dict[str, Any]:
    """Load simulation presets from config/simulations.yaml.

    Expected structure (flexible, only keys used are read):
    
    system_name:
      default:
        initial_state: [..]
        initial_states: [[..], [..]]
        presets:
          NAME:
            initial_states: [[..], [..]]
      CONTROL_NAME:
        ...

    Falls back to 'default' entry if the specific control_name is not found.
    Returns a dict which may include keys like 'initial_state', 'initial_states', 'presets'.
    If preset_name is provided and found under 'presets', its dict is returned; else the
    immediate node under (system_name, control_name or default) is returned.
    """
    cfg_path = Path(_SIM_CONFIG_OVERRIDE_PATH or path)
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f) or {}
    sys_node = cfg.get(system_name, {})
    # Try specific control name first, fall back to default
    if control_name in sys_node:
        ctrl_node = sys_node[control_name]
    else:
        ctrl_node = sys_node.get('default', {})
    
    if preset_name is None:
        return ctrl_node
    return ctrl_node.get('presets', {}).get(preset_name, {})


def _merge_shallow(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge two dictionaries, with override taking precedence.

    This does not deep-merge nested dicts except at the top level; that's sufficient
    for our simulations config where presets typically override top-level keys like
    dt, steps, fps, initial_state(s), and uncertainty_grid_value_tag.
    """
    out = dict(base or {})
    for k, v in (override or {}).items():
        out[k] = v
    return out


def _merge_deep(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries.

    Dict values are merged recursively, scalars/sequences are replaced by override.
    Returns a new dict (does not mutate inputs).
    """
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_deep(out[k], v)
        else:
            out[k] = v
    return out


def load_simulation_config(
    system_name: str,
    control_name: Optional[str] = None,
    preset_name: Optional[str] = None,
    *,
    path: str = 'config/simulations.yaml',
) -> Dict[str, Any]:
    """Load simulation config with optional preset overlay.

    Selection order mirrors other config loaders:
      - Take node at config[system_name][control_name] if present, else 'default'.
      - If preset_name is provided, merge that preset node on top (shallow override).

    Returns {} if file or keys are missing.
    """
    base = load_simulation_presets(system_name, (control_name or 'default'), None, path=path)
    if preset_name:
        preset = load_simulation_presets(system_name, (control_name or 'default'), preset_name, path=path)
        return _merge_shallow(base, preset)
    return base


def load_nn_training_config(
    system_name: str,
    preset: Optional[str] = None,
    *,
    path: str = 'config/nn_training.yaml',
) -> Dict[str, Any]:
    """Load NN training config for a system with optional preset overlay (deep merge).

    Expected structure:
      system_name:
        default:
          ...
          presets:
            NAME: {...}

    Returns {} if file or keys are missing.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f) or {}
    sys_node = cfg.get(system_name, {})
    base = dict(sys_node.get('default', {}))
    if not preset:
        return base
    presets = (base or {}).get('presets', {})
    overlay = presets.get(preset, {}) if isinstance(presets, dict) else {}
    return _merge_deep(base, overlay)


def load_monte_carlo_config(
    system_name: str,
    preset_name: Optional[str] = None,
    *,
    path: str = 'config/monte_carlo_value.yaml',
) -> Dict[str, Any]:
    """Load Monte Carlo config for a system with optional preset overlay.

    Expected structure:
      system_name:
        default: { slices: {vary_dims: [i,j], fixed: {k: val, ...}}, grid_resolution: [nx, ny], dt: 0.05, total_samples_per_state: N, snapshot_samples: [..] }
        PRESET_NAME: {...}

    Returns {} if file or keys are missing.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f) or {}
    sys_node = cfg.get(system_name, {})
    base = dict(sys_node.get('default', {}))
    if not preset_name:
        return base
    overlay = sys_node.get(preset_name, {}) if isinstance(sys_node, dict) else {}
    # Shallow override is sufficient
    return _merge_shallow(base, overlay)
