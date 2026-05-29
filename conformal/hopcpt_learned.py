from __future__ import annotations
from typing import Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("HopCPTLearned requires PyTorch.") from e

from .base import BaseConformalizer, quantile_index_bounds


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, p: float) -> float:
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    s = w.sum()
    if s <= 0 or np.all(w == 0):
        return float(np.quantile(v, p))
    cdf = (w.cumsum() - 0.5 * w) / s
    return float(np.interp(p, cdf, v))


class _Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, cosine: bool = True):
        super().__init__()
        self.cosine = cosine
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.Wq = nn.Linear(out_dim, out_dim, bias=False)
        self.Wk = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        if self.cosine:
            z = F.normalize(z, dim=1, eps=1e-8)
        q = self.Wq(z)
        k = self.Wk(z)
        if self.cosine:
            q = F.normalize(q, dim=1, eps=1e-8)
            k = F.normalize(k, dim=1, eps=1e-8)
        return z, q, k


class HopCPTLearnedConformalizer(BaseConformalizer):
    """
    Learns to reconstruct conformity vectors S_t from other calibration vectors
    via Hopfield/attention weights. Loss: (1/T) * sum_t || S_t - sum_{j≠t} a_{t→j} S_j ||^2
    """

    def __init__(
        self,
        levels: Optional[List[float]] = None,
        hidden_dim: int = 64,
        emb_dim: int = 64,
        beta: float = 20.0,
        topk: Optional[int] = None,
        cosine: bool = True,
        epochs: int = 25,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: Optional[str] = None,
        n_epochs: Optional[int] = None,
        **kw,
    ):
        if n_epochs is not None:
            epochs = int(n_epochs)
        base_keys = ("alpha_values", "value_bounds", "enforce_monotonic")
        base_kwargs = {k: kw.pop(k) for k in list(kw.keys()) if k in base_keys}
        super().__init__(**base_kwargs)

        self.levels = levels if levels is not None else [0.9, 0.8, 0.7, 0.6]
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.beta = float(beta)
        self.topk = topk
        self.cosine = cosine
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder: Optional[_Encoder] = None
        self._S_cal: Optional[np.ndarray] = None
        self._K_mem: Optional[np.ndarray] = None
        self._cov_dim: Optional[int] = None
        self._li_ui_by_level = {}

    def _build_conformity_matrix(self, actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
        T = actual.shape[0]
        A = len(self.levels)
        S = np.empty((T, A), dtype=float)
        self._li_ui_by_level = {}
        for j, c in enumerate(self.levels):
            li, ui = quantile_index_bounds(c)
            self._li_ui_by_level[c] = (li, ui)
            S[:, j] = np.maximum(forecast[li, :] - actual, actual - forecast[ui, :])
        return S

    def _compute_weights(self, q: np.ndarray, K: np.ndarray) -> np.ndarray:
        logits = self.beta * (K @ q)
        logits -= logits.max()
        w = np.exp(logits)
        if self.topk is not None and self.topk < w.size:
            idx = np.argpartition(-w, self.topk)[: self.topk]
            mask = np.zeros_like(w, dtype=bool)
            mask[idx] = True
            w = w * mask
        s = w.sum()
        return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

    def _fit_impl(self) -> None:
        fd = self.fit_data
        y, Q, C = fd.actual, fd.forecast, fd.cov_past
        if C is None:
            raise ValueError("HopCPTLearned requires covariates (past_cov).")

        S = self._build_conformity_matrix(y, Q)
        self._S_cal = S.copy()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        T, D = C.shape
        self._cov_dim = D

        X = torch.tensor(C, dtype=torch.float32, device=self.device)
        S_t = torch.tensor(S, dtype=torch.float32, device=self.device)

        self.encoder = _Encoder(D, self.hidden_dim, self.emb_dim, cosine=self.cosine).to(self.device)
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.epochs):
            self.encoder.train()
            _, Qh, Kh = self.encoder(X)
            logits = self.beta * (Qh @ Kh.t())
            logits.fill_diagonal_(float("-inf"))

            if self.topk is not None and self.topk < T:
                kth = torch.topk(logits, k=self.topk, dim=1).values[:, -1].unsqueeze(1)
                keep = logits >= kth
                logits = torch.where(keep, logits, torch.full_like(logits, float("-inf")))

            A_mat = F.softmax(logits, dim=1)
            S_hat = A_mat @ S_t
            loss = F.mse_loss(S_hat, S_t, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.encoder.eval()
        with torch.no_grad():
            _, _, Kh = self.encoder(X)
            self._K_mem = Kh.detach().cpu().numpy()

    def update(self, forecast_q: np.ndarray, point_cov: Optional[np.ndarray] = None) -> np.ndarray:
        if point_cov is None:
            raise ValueError("HopCPTLearned.update requires point_cov.")
        if self.encoder is None or self._K_mem is None or self._S_cal is None:
            raise RuntimeError("Call fit() before update().")

        pc = np.asarray(point_cov).reshape(1, -1)
        assert pc.shape[1] == self._cov_dim, "point_cov dim mismatch"
        with torch.no_grad():
            x = torch.tensor(pc, dtype=torch.float32, device=self.device)
            _, qh, _ = self.encoder(x)
            q = qh.detach().cpu().numpy().reshape(-1)

        w = self._compute_weights(q, self._K_mem)

        out = forecast_q.copy()
        for c in self.levels:
            li, ui = self._li_ui_by_level[c]
            j = self.levels.index(c)
            delta = _weighted_quantile(self._S_cal[:, j], w, p=c)
            out[li] -= delta
            out[ui] += delta
        return out

    def validation_recon_loss(
        self,
        y_val: np.ndarray,
        Q_val: np.ndarray,
        cov_val: np.ndarray,
    ) -> float:
        if self.encoder is None or self._K_mem is None or self._S_cal is None:
            raise RuntimeError("HopCPT must be fit() before validation_recon_loss().")
        if cov_val is None:
            raise ValueError("cov_val is required for HopCPT validation loss.")

        S_val = self._build_conformity_matrix(y_val, Q_val)
        T_val, A = S_val.shape

        with torch.no_grad():
            Xv = torch.tensor(cov_val, dtype=torch.float32, device=self.device)
            _, Qv, _ = self.encoder(Xv)
            Qv_np = Qv.detach().cpu().numpy()

        K = self._K_mem
        S_cal = self._S_cal

        se_sum = 0.0
        for t in range(T_val):
            w = self._compute_weights(Qv_np[t], K)
            S_hat_t = w @ S_cal
            diff = S_hat_t - S_val[t]
            se_sum += float(np.dot(diff, diff))

        return se_sum / (T_val * A)
