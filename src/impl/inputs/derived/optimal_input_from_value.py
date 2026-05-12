"""Expose an optimised channel from a cached value function as an Input."""

from __future__ import annotations

from typing import Literal, Type
import numpy as np

import torch

from src.core.hj_reachability import ChannelMode
from src.core.inputs import Input
from src.core.systems import System
from src.core.values import Value
from src.utils.registry import get_system_class

ChannelLiteral = Literal["control", "disturbance", "uncertainty"]

__all__ = ["OptimalInputFromValue"]


class OptimalInputFromValue(Input):
    """Adapter that converts optimised Value channels into Input objects."""

    system_class: Type[System] = System
    type: ChannelLiteral = "uncertainty"
    time_invariant: bool = False

    def __init__(self, value: Value, *, channel: ChannelLiteral = "uncertainty") -> None:
        if not isinstance(value, Value):
            raise TypeError("OptimalInputFromValue expects a Value instance.")

        self._value = value
        self._channel = channel

        hj_dynamics = getattr(value, "hj_dynamics", None)
        if hj_dynamics is None:
            raise ValueError("Value must expose hj_dynamics with solver context.")

        cfg = getattr(hj_dynamics, channel, None)
        if cfg is None:
            raise ValueError(f"HJ dynamics do not expose channel '{channel}'.")
        if cfg.mode != ChannelMode.OPTIMIZE:
            raise ValueError(f"Channel '{channel}' is not optimised in the supplied value function.")

        self.type = channel
        self.time_invariant = getattr(value, "time_invariant", False)
        self._hj_dynamics = hj_dynamics
        self._system: System | None = None
        self.dim: int | None = None

        inferred = self._infer_system_class()
        if inferred is not None:
            self.system_class = inferred

    # ------------------------------------------------------------------ #
    # Input interface
    # ------------------------------------------------------------------ #
    def bind(self, system: System) -> None:
        if not isinstance(system, self.system_class):
            raise TypeError(f"{type(system).__name__} is incompatible with {self.system_class.__name__}.")

        self._system = system
        if self._channel == "control":
            self.dim = system.control_dim
        elif self._channel == "disturbance":
            self.dim = system.disturbance_dim
        elif self._channel == "uncertainty":
            self.dim = system.state_dim

    def input(self, state: torch.Tensor, time: float) -> torch.Tensor:
        if self._system is None:
            raise RuntimeError("bind() must be called before using OptimalInputFromValue.")

        # Enforce time within saved GridValue range [t_min, t_max]
        times = getattr(self._value, 'times', None)
        if times is None:
            raise RuntimeError("Value has no time axis available for range checking.")
        t_arr = np.asarray(times, dtype=float)
        if t_arr.size == 0:
            raise RuntimeError("Value has an empty time axis.")
        t_min = float(np.min(t_arr))
        t_max = float(np.max(t_arr))
        t = float(time)
        if not (t_min <= t <= t_max):
            raise ValueError(f"Time {t:.6f} is outside the value time domain [{t_min:.6f}, {t_max:.6f}].")

        state = torch.as_tensor(state)
        if self._channel == "uncertainty":
            result = self._value.optimal_uncertainty(state, time)
            if result is None:
                grad = self._value.gradient(state, time)
                low, high = self._system.uncertainty_limits(state, time)
                low = torch.as_tensor(low, dtype=state.dtype, device=state.device)
                high = torch.as_tensor(high, dtype=state.dtype, device=state.device)
                result = torch.where(grad >= 0, low, high)
        elif self._channel == "control":
            result = self._value.optimal_control(state, time)
        else:  # disturbance
            result = self._value.optimal_disturbance(state, time)

        if result is None:
            raise NotImplementedError(f"No optimal {self._channel} extractor is available.")
        return torch.as_tensor(result, dtype=state.dtype, device=state.device)

    def to(self, device: torch.device | str):  # pragma: no cover - mirrors torch API
        return self

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _infer_system_class(self) -> Type[System] | None:
        sys_attr = getattr(self._hj_dynamics, "system", None)
        if isinstance(sys_attr, type) and issubclass(sys_attr, System):
            return sys_attr
        if isinstance(sys_attr, System):
            return type(sys_attr)

        sys_instance = getattr(self._hj_dynamics, "system_instance", None)
        if isinstance(sys_instance, System):
            return type(sys_instance)

        system_name = getattr(self._value, "metadata", {}).get("system") or getattr(self._value, "system_name", None)
        if not system_name:
            return None

        cls = get_system_class(system_name)
        return cls
        return None
