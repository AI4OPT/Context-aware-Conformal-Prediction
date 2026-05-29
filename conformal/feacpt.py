from __future__ import annotations
from typing import Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("FEACPTConformalizer requires PyTorch.") from e

from .base import BaseConformalizer, quantile_index_bounds
from eval.metrics import winkler_score


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, p: float) -> float:
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    s = w.sum()
    if s <= 0 or np.all(w == 0):
        return float(np.quantile(v, p))
    cdf = (w.cumsum() - 0.5 * w) / s
    return float(np.interp(p, cdf, v))


def _build_posenc(T: int) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    x = t / max(T - 1, 1)
    s = np.sin(2.0 * np.pi * x)
    c = np.cos(2.0 * np.pi * x)
    ones = np.ones_like(x)
    return np.stack([x, s, c, ones], axis=1)


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


class FEACPTConformalizer(BaseConformalizer):
    """
    Frequency-Enhanced Attention (FEA) Conformalizer.

    Uses FFT-based attention over calibration keys to assign weights,
    then applies weighted conformity quantiles per coverage level.
    Falls back to time-position encoding when covariates are absent.
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
        seed: Optional[int] = None,
        device: Optional[str] = None,
        fea_modes: int = 64,
        fea_random: bool = True,
        fea_activation: str = "softmax",
        fea_loss: str = "reconstruction",
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
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.fea_modes = int(fea_modes)
        self.fea_random = bool(fea_random)
        self.fea_activation = fea_activation.lower()
        self.fea_loss = fea_loss.lower()
        assert self.fea_activation in ("softmax", "tanh")
        assert self.fea_loss in ("reconstruction", "winkler")

        self.encoder: Optional[_Encoder] = None
        self._S_cal: Optional[np.ndarray] = None
        self._K_mem_f: Optional[np.ndarray] = None
        self._cov_dim: Optional[int] = None
        self._li_ui_by_level = {}
        self._fea_idx_cpu: Optional[torch.Tensor] = None

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

    def _init_fea_indices(self, P: int) -> None:
        rlen = P // 2 + 1
        M = min(self.fea_modes, rlen)
        rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()
        if self.fea_random:
            if M <= 1:
                idx = np.array([0], dtype=np.int64)
            else:
                pool = np.arange(1, rlen, dtype=np.int64)
                take = rng.choice(pool, size=M - 1, replace=False)
                idx = np.sort(np.concatenate([np.array([0], dtype=np.int64), take]))
        else:
            idx = np.arange(M, dtype=np.int64)
        self._fea_idx_cpu = torch.from_numpy(idx)

    @staticmethod
    def _rfft_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        Xf = torch.fft.rfft(x, dim=1)
        return Xf.index_select(dim=1, index=idx)

    def _fea_logits_matrix(self, Qh: torch.Tensor, Kh: torch.Tensor) -> torch.Tensor:
        T, P = Qh.shape
        if self._fea_idx_cpu is None:
            self._init_fea_indices(P)
        idx = self._fea_idx_cpu.to(Qh.device)
        Qf = self._rfft_select(Qh, idx)
        Kf = self._rfft_select(Kh, idx)
        gram = Qf @ Kf.conj().transpose(0, 1)
        return self.beta * torch.real(gram).to(torch.float32)

    def _fea_logits_single(self, qh: torch.Tensor, Kf_mem: torch.Tensor) -> torch.Tensor:
        if self._fea_idx_cpu is None:
            self._init_fea_indices(qh.shape[1])
        idx = self._fea_idx_cpu.to(qh.device)
        qf = self._rfft_select(qh, idx)
        return self.beta * torch.real(qf @ Kf_mem.conj().transpose(0, 1)).squeeze(0)

    def _fit_impl(self) -> None:
        fd = self.fit_data
        y, Q, C = fd.actual, fd.forecast, fd.cov_past

        S = self._build_conformity_matrix(y, Q)
        self._S_cal = S.copy()

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        T = y.shape[0]

        if C is None:
            C = _build_posenc(T)
        else:
            C = np.asarray(C, dtype=np.float32)

        self._cov_dim = int(C.shape[1])

        X = torch.tensor(C, dtype=torch.float32, device=self.device)
        S_t = torch.tensor(S, dtype=torch.float32, device=self.device)

        self.encoder = _Encoder(self._cov_dim, self.hidden_dim, self.emb_dim, cosine=self.cosine).to(self.device)
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._init_fea_indices(self.emb_dim)

        for _ in range(self.epochs):
            self.encoder.train()
            _, Qh, Kh = self.encoder(X)
            logits = self._fea_logits_matrix(Qh, Kh)
            logits.fill_diagonal_(float("-inf"))

            if self.topk is not None and self.topk < T:
                kth = torch.topk(logits, k=self.topk, dim=1).values[:, -1].unsqueeze(1)
                keep = logits >= kth
                logits = torch.where(keep, logits, torch.full_like(logits, float("-inf")))

            if self.fea_activation == "tanh":
                A_mat = F.softmax(torch.tanh(logits), dim=1)
            else:
                A_mat = F.softmax(logits, dim=1)

            if self.fea_loss == "winkler":
                A_np = A_mat.detach().cpu().numpy()
                pred_q = np.empty((99, T))
                for t in range(T):
                    w = A_np[t]
                    for c_idx, c in enumerate(self.levels):
                        li, ui = self._li_ui_by_level[c]
                        j = self.levels.index(c)
                        delta = _weighted_quantile(self._S_cal[:, j], w, p=c)
                        pred_q[li, t] = Q[t, li] - delta
                        pred_q[ui, t] = Q[t, ui] + delta
                winkler_scores = winkler_score(y, pred_q.T)
                loss = torch.tensor(
                    np.mean(list(winkler_scores.values())),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                S_hat = A_mat @ S_t
                loss = F.mse_loss(S_hat, S_t, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.encoder.eval()
        with torch.no_grad():
            _, _, Kh = self.encoder(X)
            idx = self._fea_idx_cpu.to(self.device)
            Kf = torch.fft.rfft(Kh, dim=1).index_select(1, idx)
            self._K_mem_f = Kf.detach().cpu().numpy()

    def update(self, forecast_q: np.ndarray, point_cov: Optional[np.ndarray] = None) -> np.ndarray:
        if self.encoder is None or self._K_mem_f is None or self._S_cal is None:
            raise RuntimeError("Call fit() before update().")

        if point_cov is None:
            w = np.full(self._K_mem_f.shape[0], 1.0 / self._K_mem_f.shape[0], dtype=np.float32)
        else:
            pc = np.asarray(point_cov).reshape(1, -1).astype(np.float32)
            assert pc.shape[1] == self._cov_dim, "point_cov dim mismatch"
            with torch.no_grad():
                x = torch.tensor(pc, dtype=torch.float32, device=self.device)
                _, qh, _ = self.encoder(x)
                Kf_mem = torch.from_numpy(self._K_mem_f).to(self.device)
                logits = self._fea_logits_single(qh, Kf_mem)
                if self.topk is not None and self.topk < logits.numel():
                    kth = torch.topk(logits, k=self.topk).values[-1]
                    mask = logits >= kth
                    logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
                if self.fea_activation == "tanh":
                    w = F.softmax(torch.tanh(logits), dim=0).detach().cpu().numpy()
                else:
                    w = F.softmax(logits, dim=0).detach().cpu().numpy()

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
        cov_val: Optional[np.ndarray] = None,
    ) -> float:
        if self.encoder is None or self._K_mem_f is None or self._S_cal is None:
            raise RuntimeError("FEACPTConformalizer must be fit() before validation_recon_loss().")

        S_val = self._build_conformity_matrix(y_val, Q_val)
        T_val, A = S_val.shape

        if cov_val is None:
            w = np.full(self._S_cal.shape[0], 1.0 / self._S_cal.shape[0], dtype=np.float32)
            S_cal = torch.tensor(self._S_cal, dtype=torch.float32)
            se = 0.0
            for t in range(T_val):
                S_hat_t = torch.tensor(w, dtype=torch.float32) @ S_cal
                diff = S_hat_t - torch.tensor(S_val[t], dtype=torch.float32)
                se += float(torch.dot(diff, diff).item())
            return se / (T_val * A)

        with torch.no_grad():
            Xv = torch.tensor(np.asarray(cov_val, dtype=np.float32), dtype=torch.float32, device=self.device)
            _, Qv, _ = self.encoder(Xv)
            Kf_mem = torch.from_numpy(self._K_mem_f).to(self.device)
            S_cal = torch.tensor(self._S_cal, dtype=torch.float32, device=self.device)

            se_sum = 0.0
            for t in range(T_val):
                qh = Qv[t : t + 1]
                logits = self._fea_logits_single(qh, Kf_mem)
                if self.topk is not None and self.topk < logits.numel():
                    kth = torch.topk(logits, k=self.topk).values[-1]
                    mask = logits >= kth
                    logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
                if self.fea_activation == "tanh":
                    w = F.softmax(torch.tanh(logits), dim=0)
                else:
                    w = F.softmax(logits, dim=0)
                S_hat_t = (w.unsqueeze(0) @ S_cal).squeeze(0)
                diff = S_hat_t - torch.tensor(S_val[t], dtype=torch.float32, device=self.device)
                se_sum += float(torch.dot(diff, diff).item())

        return se_sum / (T_val * A)
