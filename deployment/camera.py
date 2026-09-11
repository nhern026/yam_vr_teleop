"""Signal filters for the deployment stack.

Reconstructed from how `quest_teleop` uses it: keyword constructor
(`min_cutoff_hz`, `beta`, optional `d_cutoff_hz`), `__call__(x, dt)`,
`reset()`, and a static `_alpha(cutoff_hz, dt)` that `OneEuroQuatFilter`
borrows.
"""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
  """One Euro filter (Casiez, Roussel, Vogel, CHI 2012) on a vector signal.

  The low-pass cutoff rises with the signal's smoothed speed:
  ``cutoff = min_cutoff_hz + beta * |dx/dt|``. A still signal is filtered at
  ``min_cutoff_hz``; a fast one passes almost untouched. Speed is the norm of
  the whole vector, matching the scalar angular speed `OneEuroQuatFilter` uses,
  so position and orientation respond to the same `beta` the same way.
  """

  def __init__(self, *, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0):
    if min_cutoff_hz <= 0.0 or beta < 0.0 or d_cutoff_hz <= 0.0:
      raise ValueError("filter cutoffs must be positive and beta non-negative")
    self._min_cutoff = float(min_cutoff_hz)
    self._beta = float(beta)
    self._d_cutoff = float(d_cutoff_hz)
    self._x: np.ndarray | None = None
    self._dx: np.ndarray | None = None

  @staticmethod
  def _alpha(cutoff_hz: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass at ``cutoff_hz`` sampled every ``dt``."""
    tau = 1.0 / (2.0 * math.pi * float(cutoff_hz))
    return 1.0 / (1.0 + tau / float(dt))

  def reset(self) -> None:
    self._x = None
    self._dx = None

  def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
      raise ValueError("dt must be positive")
    x = np.asarray(x, dtype=np.float64)
    if self._x is None or self._dx is None:
      self._x = x.copy()
      self._dx = np.zeros_like(x)
      return x.copy()
    # Derivative against the previous *filtered* value, as in the paper.
    dx = (x - self._x) / dt
    self._dx = self._dx + self._alpha(self._d_cutoff, dt) * (dx - self._dx)
    cutoff = self._min_cutoff + self._beta * float(np.linalg.norm(self._dx))
    self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
    return self._x.copy()
