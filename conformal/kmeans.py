import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .base import BaseConformalizer, quantile_index_bounds


class KMeansConformalizer(BaseConformalizer):
    def __init__(self, n_clusters=8, **kw):
        super().__init__(**kw)
        self.n_clusters = int(n_clusters)
        self.scaler = MinMaxScaler()
        self.kmeans: KMeans | None = None
        self.labels_: np.ndarray | None = None
        self.Xref: np.ndarray | None = None

    def _fit_impl(self):
        if self.fit_data.cov_past is None:
            raise ValueError("KMeansConformalizer requires past covariates.")
        X = self.fit_data.cov_past
        Xs = self.scaler.fit_transform(X)
        km = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=0)
        labels = km.fit_predict(Xs)
        self.kmeans = km
        self.labels_ = labels
        self.Xref = Xs

        A, F = self.fit_data.actual, self.fit_data.forecast
        self._scores_by_level = {}
        self._li_ui_by_level = {}
        for c in self.alpha_values:
            level = np.round(1 - c, 2)
            li, ui = quantile_index_bounds(level)
            self._li_ui_by_level[level] = (li, ui)
            self._scores_by_level[level] = np.maximum(F[li] - A, A - F[ui])

    def weights_for_point(self, point_cov: np.ndarray) -> np.ndarray:
        if self.kmeans is None or self.labels_ is None or self.Xref is None:
            raise RuntimeError("Call fit() before weights_for_point().")
        x = self.scaler.transform(point_cov.reshape(1, -1))
        cl = int(self.kmeans.predict(x)[0])
        T = self.Xref.shape[0]
        w = np.zeros(T, dtype=float)
        idx = np.where(self.labels_ == cl)[0]
        if idx.size:
            w[idx] = 1.0 / idx.size
        return w

    @staticmethod
    def _weighted_quantile(values: np.ndarray, q: float, w: np.ndarray) -> float:
        order = np.argsort(values)
        v = values[order]
        ww = w[order]
        cdf = np.cumsum(ww) - 0.5 * ww
        cdf /= cdf[-1]
        return float(np.interp(q, cdf, v))

    def update(self, forecast_q: np.ndarray, point_cov=None) -> np.ndarray:
        if point_cov is None:
            raise ValueError("KMeansConformalizer.update requires point_cov.")
        w = self.weights_for_point(np.asarray(point_cov))
        for c in self.alpha_values:
            level = np.round(1 - c, 2)
            li, ui = self._li_ui_by_level[level]
            scores = self._scores_by_level[level]
            delta = self._weighted_quantile(scores, level, w)
            forecast_q[li] -= delta
            forecast_q[ui] += delta
        return forecast_q
