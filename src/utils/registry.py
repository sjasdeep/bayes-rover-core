"""Registry and discovery utilities for systems, inputs, dynamics, and cache tags.

Consolidates class discovery and common listings
used by CLI scripts (GridInput, GridSet, GridValue).
"""
from __future__ import annotations

import functools
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type

__all__ = [
    "discover_classes",
    "get_available_system_classes",
    "get_available_input_classes",
    "get_available_hj_dynamics_classes",
    "get_system_to_inputs_map",
    "get_system_class",
    "instantiate_system",
    "get_input_class",
    "instantiate_input",
    "get_hj_dynamics_class",
    "list_grid_input_tags",
    "list_grid_set_tags",
    "list_grid_value_tags",
    "list_simulation_tags",
    "list_monte_carlo_value_tags",
]

# ----- Class discovery -----

def discover_classes(
    search_dir: str,
    package_root: str = 'src',
    filter_fn: Callable[[Type], bool] = lambda cls: True,
) -> List[Type]:
    """
    Recursively discover Python classes defined within .py files under `search_dir`,
    importing modules via canonical package names rooted at `package_root`.

    Notes
    -----
    - Ensures the parent of `package_root` is on sys.path.
    - Computes module names relative to `package_root`,
      so it works whether `search_dir` == `package_root` or any subdirectory of `package_root`.
    """
    search_dir = os.path.abspath(search_dir)

    # Locate the absolute path to `package_root`
    pkg_dir = search_dir
    while True:
        if os.path.basename(pkg_dir) == package_root:
            break
        parent = os.path.dirname(pkg_dir)
        if parent == pkg_dir:
            raise ValueError(
                f'Could not find package_root "{package_root}" above "{search_dir}".'
            )
        pkg_dir = parent

    # Ensure the parent of `package_root` is on sys.path
    parent_of_pkg = os.path.dirname(pkg_dir)
    if parent_of_pkg not in sys.path:
        sys.path.insert(0, parent_of_pkg)

    discovered: List[Type] = []

    # Walk `search_dir` but compute module names relative to `pkg_dir`
    for dirpath, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if not filename.endswith('.py') or filename == '__init__.py':
                continue
            file_path = os.path.join(dirpath, filename)
            rel_to_pkg = os.path.relpath(file_path, pkg_dir)
            mod_path = os.path.splitext(rel_to_pkg)[0].replace(os.sep, '.')
            full_name = f'{package_root}.{mod_path}'
            module = importlib.import_module(full_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__ and filter_fn(obj):
                    discovered.append(obj)
    return discovered


# ----- Convenience getters for project types -----

from src.core.inputs import Input
from src.core.systems import System
from src.impl.hj_reachability.base import HJSolverDynamics


@functools.lru_cache(maxsize=None)
def _system_classes() -> Tuple[Type[System], ...]:
    classes = discover_classes('src/impl', 'src', lambda cls: inspect.isclass(cls) and issubclass(cls, System))
    return tuple(classes)


@functools.lru_cache(maxsize=None)
def _input_classes() -> Tuple[Type[Input], ...]:
    classes = discover_classes('src/impl', 'src', lambda cls: inspect.isclass(cls) and issubclass(cls, Input))
    return tuple(classes)


@functools.lru_cache(maxsize=None)
def _hj_dynamics_classes() -> Tuple[Type[HJSolverDynamics], ...]:
    classes = discover_classes('src/impl', 'src', lambda cls: inspect.isclass(cls) and issubclass(cls, HJSolverDynamics))
    return tuple(classes)


def get_available_system_classes() -> List[Type[System]]:
    return list(_system_classes())


def get_available_input_classes() -> List[Type[Input]]:
    return list(_input_classes())


def get_available_hj_dynamics_classes() -> List[Type[HJSolverDynamics]]:
    uniq: List[Type[HJSolverDynamics]] = []
    seen: set[int] = set()
    for cls in _hj_dynamics_classes():
        if cls is HJSolverDynamics or inspect.isabstract(cls):
            continue
        # Skip private base classes (names starting with _)
        if cls.__name__.startswith('_'):
            continue
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        uniq.append(cls)
    return uniq


def get_system_to_inputs_map() -> Dict[Type[System], List[Type[Input]]]:
    inputs = [cls for cls in _input_classes() if cls.__name__ != 'GridInput']
    mapping: Dict[Type[System], List[Type[Input]]] = {}
    for system_cls in _system_classes():
        compat = [inp for inp in inputs if hasattr(inp, 'system_class') and issubclass(system_cls, inp.system_class)]
        mapping[system_cls] = compat
    return mapping


def get_system_class(name: str) -> Type[System] | None:
    return next((cls for cls in _system_classes() if cls.__name__ == name), None)


def instantiate_system(name: str) -> System:
    cls = get_system_class(name)
    if cls is None:
        raise ValueError(f"System class not found: {name}")
    return cls()


def get_input_class(name: str) -> Type[Input] | None:
    return next((cls for cls in _input_classes() if cls.__name__ == name), None)


def instantiate_input(name: str, system: System | None = None, **kwargs) -> Input:
    cls = get_input_class(name)
    if cls is None:
        raise ValueError(f"Input class not found: {name}")
    instance = cls(**kwargs)
    if system is not None:
        instance.bind(system)
    return instance


def get_hj_dynamics_class(name: str) -> Type[HJSolverDynamics] | None:
    return next((cls for cls in _hj_dynamics_classes() if cls.__name__ == name), None)


# ----- Cache tag listings -----

def list_grid_input_tags() -> List[str]:
    d = Path('.cache') / 'grid_inputs'
    return sorted([p.stem for p in d.glob('*.pkl')]) if d.exists() else []


def list_grid_set_tags() -> List[str]:
    d = Path('.cache') / 'grid_sets'
    return sorted([p.stem for p in d.glob('*.pkl')]) if d.exists() else []


def list_grid_value_tags() -> List[str]:
    d = Path('.cache') / 'grid_values'
    return sorted([p.stem for p in d.glob('*.pkl')]) if d.exists() else []


def list_simulation_tags() -> List[str]:
    """List all simulation result tags (directories with results.pkl)."""
    d = Path('outputs') / 'simulations'
    if not d.exists():
        return []
    return sorted([p.name for p in d.iterdir() if p.is_dir() and (p / 'results.pkl').exists()])


def list_monte_carlo_value_tags() -> List[str]:
    """List all Monte Carlo value cache tags (files under .cache/monte_carlo_values)."""
    d = Path('.cache') / 'monte_carlo_values'
    return sorted([p.stem for p in d.glob('*.pkl')]) if d.exists() else []
