from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from .base import BaseConformalizer, quantile_index_bounds


class AdaptiveConformalizer(BaseConformalizer):
    def __init__(
        self,
        target_levels: Optional[List[float]] = None,
        gamma: float = 1e-4,
        alpha_min: float = 1e-6,
        alpha_max: float = 0.49,
        cover_window: int = 1,
        **kw,
    ):
        super().__init__(**kw)
        self.levels = target_levels if target_levels is not None else [0.9, 0.8, 0.7, 0.6]
        self.alpha_targets: Dict[float, float] = {c: 1 - c for c in self.levels}
        self.gamma = float(gamma)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.cover_window = int(max(1, cover_window))

        self._scores_by_level: Dict[float, np.ndarray] = {}
        self._li_ui_by_level: Dict[float, Tuple[int, int]] = {}
        self._alpha_t: Dict[float, float] = {}
        self._cover_hist: Dict[float, deque] = {}

    def _apply_bounds_and_monotonic(self, q: np.ndarray) -> np.ndarray:
        out = q
        if self.value_bounds is not None:
            lo, hi = self.value_bounds
            out = np.clip(out, lo, hi)
        if getattr(self, "enforce_monotonic", False):
            out = np.maximum.accumulate(out)
        return out

    def _fit_impl(self):
        A, F = self.fit_data.actual, self.fit_data.forecast
        for c in self.levels:
            li, ui = quantile_index_bounds(c)
            self._li_ui_by_level[c] = (li, ui)
            self._scores_by_level[c] = np.maximum(F[li] - A, A - F[ui])

    def batch_forecast(
        self,
        forecasts: np.ndarray,
        future_cov: Optional[np.ndarray] = None,
        actual_future: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        Q, Tt = forecasts.shape
        out = np.empty((Tt, Q), dtype=float)

        self._alpha_t = {c: self.alpha_targets[c] for c in self.levels}
        self._cover_hist = {c: deque(maxlen=self.cover_window) for c in self.levels}

        for t in range(Tt):
            qvec = forecasts[:, t].copy()
            intervals = {}

            for c in self.levels:
                li, ui = self._li_ui_by_level[c]
                alpha_t = float(self._alpha_t[c])
                q = np.quantile(self._scores_by_level[c], 1.0 - alpha_t)
                qvec[li] -= q
                qvec[ui] += q

            qvec = self._apply_bounds_and_monotonic(qvec)

            for c in self.levels:
                li, ui = self._li_ui_by_level[c]
                intervals[c] = (qvec[li], qvec[ui])

            out[t] = qvec

            if actual_future is not None:
                y = float(actual_future[t])
                for c in self.levels:
                    L, U = intervals[c]
                    covered = 1.0 if (y >= L and y <= U) else 0.0
                    hist = self._cover_hist[c]
                    hist.append(covered)
                    coverage_t = float(np.mean(hist)) if self.cover_window > 1 else covered
                    error_rate = 1.0 - coverage_t
                    new_alpha = self._alpha_t[c] + self.gamma * (self.alpha_targets[c] - error_rate)
                    self._alpha_t[c] = float(np.clip(new_alpha, self.alpha_min, self.alpha_max))

        return out

    def update(self, forecast_q: np.ndarray, point_cov=None) -> np.ndarray:
        q = forecast_q.copy()
        for c in self.levels:
            li, ui = self._li_ui_by_level[c]
            alpha_t = float(self._alpha_t.get(c, self.alpha_targets[c]))
            adj = np.quantile(self._scores_by_level[c], 1.0 - alpha_t)
            q[li] -= adj
            q[ui] += adj
        return self._apply_bounds_and_monotonic(q)
