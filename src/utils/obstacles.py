"""2D obstacle primitives with signed distance functions for collision checking."""
from __future__ import annotations

from typing import List

import torch

__all__ = ["Box2D", "Circle2D", "signed_distance_to_obstacles", "draw_obstacles_2d"]


def _ensure_batch(p: torch.Tensor) -> torch.Tensor:
    """Accept [2] or [N,2]; return [N,2]."""
    if p.ndim == 1:
        return p.unsqueeze(0)
    return p

class Box2D:
    def __init__(self, center, rotation, length, width, dtype=torch.float32):
        """
        center: (2,) array-like
        rotation: float (radians, CCW)
        length, width: floats (box extents along local x/y)
        Notes:
          - Internally stores parameters as tensors without fixed device.
          - At query, parameters are cast to the point's device/dtype.
        """
        self.center = torch.as_tensor(center, dtype=dtype)
        self.rotation = torch.as_tensor(rotation, dtype=dtype)
        self.half_size = torch.as_tensor([length/2, width/2], dtype=dtype)

    def signed_distance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Signed Euclidean distance to the box boundary.
        Negative inside, positive outside, zero on boundary.
        Accepts [2] or [N,2]; returns [N].
        """
        p = torch.as_tensor(point)
        p = _ensure_batch(p)

        # Cast params to match the query point's device/dtype
        dtype, device = p.dtype, p.device
        c = self.center.to(device=device, dtype=dtype)
        h = self.half_size.to(device=device, dtype=dtype)
        ang = self.rotation.to(device=device, dtype=dtype)

        # Rotation (world -> box-local)
        cos, sin = torch.cos(-ang), torch.sin(-ang)
        R = torch.stack((torch.stack((cos, -sin)), torch.stack((sin, cos))))  # [2,2]

        local = (R @ (p - c).T).T  # [N,2]
        d = torch.abs(local) - h   # [N,2]

        outside = torch.linalg.norm(torch.clamp(d, min=0), dim=1)
        inside = torch.clamp(torch.max(d, dim=1).values, max=0)
        return outside + inside  # [N]


class Circle2D:
    def __init__(self, center, radius, dtype=torch.float32):
        """
        center: (2,) array-like
        radius: float
        Notes:
          - Parameters are cast to the query point's device/dtype at call time.
        """
        self.center = torch.as_tensor(center, dtype=dtype)
        self.radius = torch.as_tensor(radius, dtype=dtype)

    def signed_distance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Signed Euclidean distance to the circle.
        Negative inside, positive outside, zero on boundary.
        Accepts [2] or [N,2]; returns [N].
        """
        p = torch.as_tensor(point)
        p = _ensure_batch(p)

        dtype, device = p.dtype, p.device
        c = self.center.to(device=device, dtype=dtype)
        r = self.radius.to(device=device, dtype=dtype)

        return torch.linalg.norm(p - c, dim=1) - r  # [N]

def signed_distance_to_obstacles(obstacles: List, points: torch.Tensor) -> torch.Tensor:
    """
    Compute signed distance to a set of obstacles (union semantics).

    Args:
        obstacles: list of objects each implementing `signed_distance(points) -> [N]`
        points: [2] or [N,2] tensor

    Returns:
        [N] tensor of signed distances (min over obstacles)
    """
    # Collect distances from all obstacles
    ds = [obs.signed_distance(points) for obs in obstacles]
    D = torch.stack(ds, dim=0)  # [K, N]
    return torch.min(D, dim=0).values  # union distance (closest obstacle)


def draw_obstacles_2d(ax, system, *, zorder: int = 100) -> None:
    """Draw 2D obstacles from a system onto a matplotlib axis.
    
    Renders Circle2D as red translucent circles and Box2D as orange translucent rectangles.
    Safe to call even if system has no obstacles attribute.
    
    Args:
        ax: matplotlib Axes to draw on
        system: System instance (may have .obstacles attribute)
        zorder: Z-order for obstacle patches (default 100, above most plots)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    if not hasattr(system, 'obstacles') or not system.obstacles:
        return
    
    for obstacle in system.obstacles:
        if isinstance(obstacle, Circle2D):
            center = obstacle.center.detach().cpu().numpy() if isinstance(obstacle.center, torch.Tensor) else np.asarray(obstacle.center)
            radius = float(obstacle.radius.detach().cpu().item() if isinstance(obstacle.radius, torch.Tensor) else float(obstacle.radius))
            circ = plt.Circle(
                center,
                radius,
                facecolor=(1.0, 0.0, 0.0, 0.3),  # red with alpha
                edgecolor='red',
                linewidth=1.2,
                zorder=zorder,
            )
            ax.add_patch(circ)
        elif isinstance(obstacle, Box2D):
            cx, cy = obstacle.center
            hx, hy = obstacle.half_size
            rot = obstacle.rotation
            cx_f = float(cx.detach().cpu().item() if isinstance(cx, torch.Tensor) else float(cx))
            cy_f = float(cy.detach().cpu().item() if isinstance(cy, torch.Tensor) else float(cy))
            hx_f = float(hx.detach().cpu().item() if isinstance(hx, torch.Tensor) else float(hx))
            hy_f = float(hy.detach().cpu().item() if isinstance(hy, torch.Tensor) else float(hy))
            rot_deg = float((rot.detach().cpu().item() if isinstance(rot, torch.Tensor) else float(rot)) * 180.0 / np.pi)
            rect = Rectangle(
                (cx_f - hx_f, cy_f - hy_f), 2*hx_f, 2*hy_f,
                angle=rot_deg,
                facecolor=(1.0, 0.647, 0.0, 0.3),  # orange with alpha
                edgecolor='orange',
                linewidth=1.2,
                zorder=zorder,
            )
            ax.add_patch(rect)