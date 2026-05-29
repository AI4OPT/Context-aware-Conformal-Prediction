from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree

from .base import BaseConformalizer, quantile_index_bounds


class KNNConformalizer(BaseConformalizer):
    def __init__(self, nneighbors: int = 200, **kw):
        super().__init__(**kw)
        self.nneighbors = int(nneighbors)
        self.scaler = MinMaxScaler()
        self.tree: KDTree | None = None
        self.Xref: np.ndarray | None = None
        self._scores_by_level: dict[float, np.ndarray] = {}
        self._li_ui_by_level: dict[float, tuple[int, int]] = {}

    def _fit_impl(self) -> None:
        if self.fit_data.cov_past is None:
            raise ValueError("KNNConformalizer requires past covariates.")
        X = self.fit_data.cov_past
        Xs = self.scaler.fit_transform(X)
        self.Xref = Xs
        self.tree = KDTree(Xs, metric="euclidean")

        A, F = self.fit_data.actual, self.fit_data.forecast
        self._scores_by_level.clear()
        self._li_ui_by_level.clear()
        for alpha in self.alpha_values:
            c = np.round(1 - alpha, 2)
            li, ui = quantile_index_bounds(c)
            self._li_ui_by_level[c] = (li, ui)
            self._scores_by_level[c] = np.maximum(F[li] - A, A - F[ui])

    def weights_for_point(self, point_cov: np.ndarray) -> np.ndarray:
        if self.tree is None or self.Xref is None:
            raise RuntimeError("Call fit() before weights_for_point().")
        x = self.scaler.transform(point_cov.reshape(1, -1))
        T = self.Xref.shape[0]
        k = max(1, min(self.nneighbors, T))
        _, idx = self.tree.query(x, k=k, return_distance=True)
        w = np.zeros(T, dtype=float)
        w[idx[0]] = 1.0 / k
        return w

    @staticmethod
    def _weighted_quantile(values: np.ndarray, q: float, w: np.ndarray) -> float:
        order = np.argsort(values)
        v, ww = values[order], w[order]
        cdf = np.cumsum(ww) - 0.5 * ww
        cdf /= cdf[-1]
        return float(np.interp(q, cdf, v))

    def update(self, forecast_q: np.ndarray, point_cov=None) -> np.ndarray:
        if point_cov is None:
            raise ValueError("KNNConformalizer.update requires point_cov.")
        w = self.weights_for_point(np.asarray(point_cov))
        for c in self.alpha_values:
            level = np.round(1 - c, 2)
            li, ui = self._li_ui_by_level[level]
            scores = self._scores_by_level[level]
            delta = self._weighted_quantile(scores, level, w)
            forecast_q[li] -= delta
            forecast_q[ui] += delta
        return forecast_q
