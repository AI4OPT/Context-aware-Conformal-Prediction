# conformal/periodic.py
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np

from .base import BaseConformalizer, quantile_index_bounds

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None  # optional; we fall back to power top-K if not present

try:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import KDTree
    from sklearn.preprocessing import MinMaxScaler
except Exception as e:
    raise ImportError("PeriodicCovariateConformalizer requires scikit-learn.") from e


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, p: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("nan")
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if not np.isfinite(s) or s <= 0 or np.all(w == 0):
        return float(np.quantile(v, p))
    idx = np.argsort(v)
    v = v[idx]; w = w[idx]
    cdf = (w.cumsum() - 0.5 * w) / s
    return float(np.interp(p, cdf, v))


def _detect_periods_fft(x: np.ndarray,
                        sampling_rate: float = 1.0,
                        n_periods: int = 3,
                        min_prominence: float = 0.05) -> List[float]:
    """
    Detect up to n_periods dominant periods via FFT. Returns periods > 2.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N < 8:
        return []

    x = x - np.nanmean(x)  # de-mean to reduce DC dominance
    fft_vals = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=sampling_rate)
    power = np.abs(fft_vals) ** 2

    pos = freqs > 0
    freqs_pos = freqs[pos]
    power_pos = power[pos]
    if freqs_pos.size == 0:
        return []

    if find_peaks is not None:
        prom = float(np.max(power_pos)) * float(min_prominence)
        peaks, props = find_peaks(power_pos, prominence=prom if np.isfinite(prom) else None)
        if peaks.size == 0:
            idx = np.argsort(-power_pos)[:n_periods]
        else:
            order = np.argsort(-props['prominences'])
            idx = peaks[order][:n_periods]
    else:
        idx = np.argsort(-power_pos)[:n_periods]

    periods = []
    for i in idx:
        f = freqs_pos[i]
        if f <= 0:
            continue
        P = 1.0 / f
        if np.isfinite(P) and P > 2.0:
            periods.append(float(P))

    # dedupe, keep order
    periods = list(dict.fromkeys([round(p, 6) for p in periods]))
    return periods[:n_periods]


def _make_embedding(t_idx: np.ndarray, periods: List[float]) -> np.ndarray:
    """
    For each t in t_idx, return [sin(2πt/P1), cos(2πt/P1), sin(2πt/P2), cos(2πt/P2), ...].
    """
    t = np.asarray(t_idx, dtype=float).reshape(-1, 1)
    if not periods:
        return np.zeros((t.shape[0], 0), dtype=float)
    feats = []
    two_pi_t = 2.0 * np.pi * t
    for P in periods:
        denom = max(P, 1e-8)
        omega_t = two_pi_t / denom
        feats.append(np.sin(omega_t))
        feats.append(np.cos(omega_t))
    return np.concatenate(feats, axis=1)


class PeriodicCovariateConformalizer(BaseConformalizer):
    """
    Build covariates from FFT-detected periodicities in calibration conformity scores,
    then use one of: backend in {'kmeans','knn','kernel'} to compute local/weighted
    conformity quantiles for each coverage level.

    No external covariates are used.
    """

    def __init__(self,
                 levels: Optional[List[float]] = None,
                 backend: str = "kernel",          # 'kmeans' | 'knn' | 'kernel'
                 n_periods: int = 3,
                 min_prominence: float = 0.05,
                 sampling_rate: float = 1.0,
                 # kmeans:
                 n_clusters: int = 5,
                 # knn:
                 nneighbors: int = 100,
                 distance_weighted: bool = True,
                 # kernel:
                 kernel: str = "rbf",              # 'rbf' | 'laplacian' | 'linear' | 'poly'
                 gamma: float = 1.0,
                 degree: int = 3,
                 coef0: float = 1.0,
                 scale_features: bool = True,
                 seed: int = 42,
                 **kw):
        """
        kw may include:
          - value_bounds=(lo, hi)
          - enforce_monotonic=True/False
          - alpha_values override, etc.
        """
        super().__init__(**kw)
        self.levels = levels if levels is not None else [0.9, 0.8, 0.7, 0.6]
        self.backend = backend.lower().strip()
        assert self.backend in {"kmeans", "knn", "kernel"}, "backend must be kmeans|knn|kernel"

        self.n_periods = int(n_periods)
        self.min_prominence = float(min_prominence)
        self.sampling_rate = float(sampling_rate)

        self.n_clusters = int(n_clusters)
        self.nneighbors = int(nneighbors)
        self.distance_weighted = bool(distance_weighted)

        self.kernel = kernel
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.scale_features = bool(scale_features)
        self.seed = int(seed)

        # storage
        self._T_cal: int = 0
        self._li_ui_by_level: Dict[float, Tuple[int, int]] = {}
        self._S_cal_by_level: Dict[float, np.ndarray] = {}
        self._periods_by_level: Dict[float, List[float]] = {}
        self._Xcal_by_level: Dict[float, np.ndarray] = {}
        self._scaler_by_level: Dict[float, MinMaxScaler] = {}

        # models per level
        self._kmeans_by_level: Dict[float, KMeans] = {}
        self._cluster_q_by_level: Dict[float, Dict[int, float]] = {}
        self._kdtree_by_level: Dict[float, KDTree] = {}

    # ----- post-process helper (clamp + monotone) -----
    def _post(self, q: np.ndarray) -> np.ndarray:
        out = q
        if getattr(self, "value_bounds", None) is not None:
            lo, hi = self.value_bounds
            out = np.clip(out, lo, hi)
        if getattr(self, "enforce_monotonic", False):
            out = np.maximum.accumulate(out)
        return out

    # ----- BaseConformalizer hooks -----
    def _fit_impl(self) -> None:
        fd = self.fit_data
        y = fd.actual            # (T_cal,)
        Q = fd.forecast          # (99, T_cal)
        self._T_cal = int(y.shape[0])

        self._li_ui_by_level.clear()
        self._S_cal_by_level.clear()
        self._periods_by_level.clear()
        self._Xcal_by_level.clear()
        self._scaler_by_level.clear()
        self._kmeans_by_level.clear()
        self._cluster_q_by_level.clear()
        self._kdtree_by_level.clear()

        t_idx = np.arange(self._T_cal, dtype=float)

        for c in self.levels:
            li, ui = quantile_index_bounds(c)
            self._li_ui_by_level[c] = (li, ui)

            S = np.maximum(Q[li] - y, y - Q[ui])   # (T_cal,)
            self._S_cal_by_level[c] = S

            periods = _detect_periods_fft(S,
                                          sampling_rate=self.sampling_rate,
                                          n_periods=self.n_periods,
                                          min_prominence=self.min_prominence)
            self._periods_by_level[c] = periods

            X = _make_embedding(t_idx, periods)    # (T_cal, 2*len(periods))
            if self.scale_features and X.shape[1] > 0:
                scaler = MinMaxScaler()
                Xs = scaler.fit_transform(X)
                self._scaler_by_level[c] = scaler
                self._Xcal_by_level[c] = Xs
            else:
                self._Xcal_by_level[c] = X

            # Backend models
            if self.backend == "kmeans":
                if X.shape[1] == 0:
                    continue  # fall back to global quantiles at inference
                km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
                labels = km.fit_predict(self._Xcal_by_level[c])
                self._kmeans_by_level[c] = km

                q_map: Dict[int, float] = {}
                for k in range(self.n_clusters):
                    m = labels == k
                    q_map[k] = float(np.quantile(S[m], c)) if np.any(m) else float(np.quantile(S, c))
                self._cluster_q_by_level[c] = q_map

            elif self.backend == "knn":
                if self._Xcal_by_level[c].shape[1] == 0:
                    continue
                self._kdtree_by_level[c] = KDTree(self._Xcal_by_level[c])

            elif self.backend == "kernel":
                # no model to fit; weights computed on the fly
                pass

    def _kernel_weights(self, Xcal: np.ndarray, x: np.ndarray) -> np.ndarray:
        if Xcal.shape[1] == 0:
            w = np.ones(Xcal.shape[0], dtype=float)
            return w / w.sum()
        if self.kernel == "rbf":
            d2 = np.sum((Xcal - x) ** 2, axis=1)
            s = np.exp(-self.gamma * d2)
        elif self.kernel == "laplacian":
            d1 = np.sum(np.abs(Xcal - x), axis=1)
            s = np.exp(-self.gamma * d1)
        elif self.kernel == "linear":
            s = (Xcal @ x.T).ravel()
            s -= s.min()
        elif self.kernel == "poly":
            s = (Xcal @ x.T).ravel()
            s = (s + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        s = np.clip(s, 0.0, None)
        tot = s.sum()
        if tot <= 0 or not np.isfinite(tot):
            s = np.ones_like(s); tot = s.sum()
        return s / tot

    def _embed_test_t(self, level: float, t_abs: float) -> np.ndarray:
        periods = self._periods_by_level.get(level, [])
        X = _make_embedding(np.array([t_abs], dtype=float), periods)  # (1, D)
        scaler = self._scaler_by_level.get(level)
        if scaler is not None and X.shape[1] > 0:
            X = scaler.transform(X)
        return X

    # ----- inference -----
    def batch_forecast(self,
                       forecasts: np.ndarray,             # (99, T_test)
                       future_cov: Optional[np.ndarray] = None,
                       actual_future: Optional[np.ndarray] = None) -> np.ndarray:
        Qn, Tt = forecasts.shape
        out = np.empty((Tt, Qn), dtype=float)

        for t in range(Tt):
            qvec = forecasts[:, t].copy()

            for c in self.levels:
                li, ui = self._li_ui_by_level[c]
                S = self._S_cal_by_level[c]
                Xcal = self._Xcal_by_level.get(c, np.zeros((S.shape[0], 0)))
                x = self._embed_test_t(c, self._T_cal + t)  # continue phase after calibration

                if Xcal.shape[1] == 0:
                    delta = float(np.quantile(S, c))
                else:
                    if self.backend == "kmeans":
                        km = self._kmeans_by_level.get(c)
                        if km is None:
                            delta = float(np.quantile(S, c))
                        else:
                            cid = int(km.predict(x)[0])
                            delta = self._cluster_q_by_level[c].get(cid, float(np.quantile(S, c)))
                    elif self.backend == "knn":
                        tree = self._kdtree_by_level.get(c)
                        if tree is None:
                            delta = float(np.quantile(S, c))
                        else:
                            k = min(self.nneighbors, Xcal.shape[0])
                            dist, idx = tree.query(x, k=k)
                            idx = idx.ravel()
                            if self.distance_weighted:
                                d = dist.ravel()
                                w = 1.0 / (d + 1e-8)
                                w = w / w.sum()
                                delta = _weighted_quantile(S[idx], w, c)
                            else:
                                delta = float(np.quantile(S[idx], c))
                    else:  # kernel
                        w = self._kernel_weights(Xcal, x)
                        delta = _weighted_quantile(S, w, c)

                qvec[li] -= delta
                qvec[ui] += delta

            out[t] = self._post(qvec)
        return out

    # optional single-step API
    def update(self, forecast_q: np.ndarray, point_cov=None) -> np.ndarray:
        q = forecast_q.copy()
        for c in self.levels:
            li, ui = self._li_ui_by_level[c]
            S = self._S_cal_by_level[c]
            delta = float(np.quantile(S, c))
            q[li] -= delta
            q[ui] += delta
        return self._post(q)
