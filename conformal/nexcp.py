import numpy as np
from .base import BaseConformalizer, quantile_index_bounds


class NExCPConformalizer(BaseConformalizer):
    def __init__(self, rho=0.98, **kw):
        super().__init__(**kw)
        self.rho = rho

    def _fit_impl(self):
        A, F = self.fit_data.actual, self.fit_data.forecast
        T = F.shape[1]
        w = self.rho ** np.arange(T - 1, -1, -1)
        w = w / w.sum()
        self.conformity_scores = {}
        for alpha in self.alpha_values:
            c = np.round(1 - alpha, 2)
            li, ui = quantile_index_bounds(c)
            scores = np.maximum(F[li] - A, A - F[ui])
            order = np.argsort(scores)
            vals, ww = scores[order], w[order]
            cdf = np.cumsum(ww)
            self.conformity_scores[alpha] = vals[np.searchsorted(cdf, c)]

    def update(self, forecast_q, point_cov=None):
        for alpha in self.alpha_values:
            c = np.round(1 - alpha, 2)
            li, ui = quantile_index_bounds(c)
            q = self.conformity_scores[alpha]
            forecast_q[li] -= q
            forecast_q[ui] += q
        return forecast_q
