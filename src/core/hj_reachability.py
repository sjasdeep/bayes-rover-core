"""Abstract interface for HJ reachability dynamics configuration.

This module defines a solver-agnostic interface for configuring Hamilton-Jacobi (HJ)
dynamics. Each signal channel (control, disturbance, uncertainty) can be configured as:

- GIVEN: A specific signal is provided (Input or Set)
- OPTIMIZE: Optimize over admissible limits from the System
- ZERO: Assume the channel is identically zero

The key semantic constraint is that optimizing over uncertainty requires the control
to be provided as a Set (capturing uncertainty-induced control variability).

Implementations should be placed in ``src/impl/hj_reachability/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from .inputs import Input
from .sets import Set
from .systems import System

__all__ = ["ChannelMode", "ChannelConfig", "HJReachabilityDynamics", "GivenKind"]

GivenKind = Literal["input", "set"]


class ChannelMode(str, Enum):
    """Mode for a signal channel in HJ dynamics.

    Attributes:
        GIVEN: A specific signal is provided (Input for nominal values or
            Set for admissible sets).
        OPTIMIZE: Optimize over admissible limits (e.g., system.control_limits()).
        ZERO: Assume the channel is identically zero.
    """

    GIVEN = "given"
    OPTIMIZE = "optimize"
    ZERO = "zero"


@dataclass
class ChannelConfig:
    """Configuration for a single channel (control, disturbance, or uncertainty).

    Attributes:
        mode: The channel mode (GIVEN, OPTIMIZE, or ZERO).
        given_kind: When mode is GIVEN, specifies whether an 'input' or 'set' is provided.
            Only the control channel supports 'set'.
        given_input: The bound Input object (when given_kind='input').
        given_set: The bound Set object (when given_kind='set').
    """

    mode: ChannelMode = ChannelMode.OPTIMIZE
    given_kind: Optional[GivenKind] = None
    given_input: Optional[Input] = None
    given_set: Optional[Set] = None

    def is_given(self) -> bool:
        """Return True if this channel has a provided Input or Set."""
        return self.mode == ChannelMode.GIVEN

    def require_given_kind(self, expected: GivenKind) -> None:
        """Assert that the channel is GIVEN with the specified kind.

        Args:
            expected: The expected given_kind ('input' or 'set').

        Raises:
            ValueError: If mode is not GIVEN or given_kind doesn't match.
        """
        if not self.is_given() or self.given_kind != expected:
            raise ValueError(
                f"Expected given kind='{expected}', but have mode={self.mode} kind={self.given_kind}"
            )


class HJReachabilityDynamics:
    """Base class for HJ reachability dynamics configuration.

    Implementations define how channels (control, disturbance, uncertainty) are
    represented, bound, and interact with HJ solvers.

    Attributes:
        system: The dynamical System this configuration is for.
        control: Configuration for the control channel.
        disturbance: Configuration for the disturbance channel.
        uncertainty: Configuration for the uncertainty channel.
    """

    system: System
    control: ChannelConfig
    disturbance: ChannelConfig
    uncertainty: ChannelConfig

    def __init__(self, system: System) -> None:
        """Initialize with default channel configs.
        
        Args:
            system: The dynamical system for this dynamics configuration.
        """
        self.system = system
        self.control = ChannelConfig()
        self.disturbance = ChannelConfig()
        self.uncertainty = ChannelConfig()

    # -------------------- Runtime class --------------------

    @classmethod
    def runtime_class(cls) -> type[HJReachabilityDynamics]:
        """Return the runtime dynamics class for simulation-time queries.
        
        By default returns this class. Override in solver-facing subclasses
        that need custom optimal_*_from_grad logic for simulation.
        """
        return cls

    # -------------------- Binding API --------------------

    def _bind_channel(self, channel: ChannelConfig, kind: str, obj) -> None:
        """Helper to bind an input or set to a channel."""
        channel.mode = ChannelMode.GIVEN
        channel.given_kind = kind
        setattr(channel, "given_input" if kind == "input" else "given_set", obj)

    def bind_control_input(self, given_input: Input) -> None:
        """Bind a nominal control Input."""
        self._bind_channel(self.control, "input", given_input)

    def bind_control_set(self, given_set: Set) -> None:
        """Bind an admissible control Set."""
        self._bind_channel(self.control, "set", given_set)

    def bind_disturbance_input(self, given_input: Input) -> None:
        """Bind a nominal disturbance Input."""
        self._bind_channel(self.disturbance, "input", given_input)

    def bind_uncertainty_input(self, given_input: Input) -> None:
        """Bind a nominal state-uncertainty Input."""
        self._bind_channel(self.uncertainty, "input", given_input)

    # -------------------- Optimal extraction --------------------

    def optimal_control_from_grad(self, state, time, grad):
        """Extract optimal control from value gradient. Override for custom logic."""
        if self.control.given_kind == "input" and self.control.given_input:
            return self.control.given_input.input(state, time)
        raise NotImplementedError(
            f"optimal_control_from_grad requires a bound control input or override in '{type(self).__name__}'."
        )

    def optimal_disturbance_from_grad(self, state, time, grad):
        """Extract optimal disturbance from value gradient. Override for custom logic."""
        if self.disturbance.given_kind == "input" and self.disturbance.given_input:
            return self.disturbance.given_input.input(state, time)
        raise NotImplementedError(
            f"optimal_disturbance_from_grad requires a bound disturbance input or override in '{type(self).__name__}'."
        )

    def optimal_uncertainty_from_grad(self, state, time, grad):
        """Extract optimal uncertainty from value gradient. Override for custom logic."""
        if self.uncertainty.given_kind == "input" and self.uncertainty.given_input:
            return self.uncertainty.given_input.input(state, time)
        raise NotImplementedError(
            f"optimal_uncertainty_from_grad requires a bound uncertainty input or override in '{type(self).__name__}'."
        )

    # -------------------- Validation --------------------

    def validate(self) -> None:
        """Validate configuration before running a solver.

        Checks:
            1. All GIVEN channels have appropriate bound objects.
            2. Bound objects are compatible with the System (type, dimension).
            3. Semantic constraint: optimizing uncertainty ⟺ control is a Set.

        Raises:
            ValueError: If any validation check fails.
        """
        # 1) Basic GIVEN binding checks per channel
        channels = (
            ("control", self.control, self.system.control_dim, "control"),
            ("disturbance", self.disturbance, self.system.disturbance_dim, "disturbance"),
            ("uncertainty", self.uncertainty, self.system.state_dim, "uncertainty"),
        )

        for name, cfg, expected_dim, expected_input_type in channels:
            if cfg.mode == ChannelMode.GIVEN:
                if cfg.given_kind == "input":
                    gi = cfg.given_input
                    if gi is None:
                        raise ValueError(f"Channel '{name}' expected an Input but is unbound")
                    # Ensure the input is compatible with this System
                    if hasattr(gi, "system_class") and gi.system_class is not None:
                        if not isinstance(self.system, gi.system_class):
                            raise ValueError(
                                f"Input for channel '{name}' expects system of type "
                                f"{gi.system_class.__name__}, but got {type(self.system).__name__}"
                            )
                    # Ensure the input type matches (or is 'any')
                    if hasattr(gi, "type") and gi.type not in ("any", expected_input_type):
                        raise ValueError(
                            f"Input for channel '{name}' has type '{gi.type}', "
                            f"expected '{expected_input_type}' or 'any'"
                        )
                    # Ensure the dimensionality matches if provided
                    if hasattr(gi, "dim") and gi.dim is not None and gi.dim != expected_dim:
                        raise ValueError(
                            f"Input for channel '{name}' has dim={gi.dim}, expected {expected_dim}"
                        )
                elif cfg.given_kind == "set":
                    if name != "control":
                        raise ValueError("Only the 'control' channel can accept a given Set")
                    gs = cfg.given_set
                    if gs is None:
                        raise ValueError("Control channel expected a Set but is unbound")
                    if hasattr(gs, "dim") and gs.dim is not None and gs.dim != self.system.control_dim:
                        raise ValueError(
                            f"Control Set has dim={gs.dim}, expected {self.system.control_dim}"
                        )
                else:
                    raise ValueError(
                        f"Channel '{name}' is GIVEN but has invalid given_kind={cfg.given_kind!r}"
                    )

        # 2) Semantic equivalence between uncertainty optimization and control Set
        control_is_set = (
            self.control.mode == ChannelMode.GIVEN and self.control.given_kind == "set"
        )
        uncertainty_is_opt = self.uncertainty.mode == ChannelMode.OPTIMIZE

        if control_is_set != uncertainty_is_opt:
            if control_is_set and not uncertainty_is_opt:
                raise ValueError(
                    "When control is provided as a Set, the uncertainty channel must be OPTIMIZE."
                )
            if uncertainty_is_opt and not control_is_set:
                raise ValueError(
                    "When uncertainty is OPTIMIZE, the control channel must be provided as a Set."
                )
