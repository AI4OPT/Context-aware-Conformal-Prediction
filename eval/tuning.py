# eval/tuning.py
from __future__ import annotations
import os, time, csv, errno, datetime, re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional, List

from conformal.factory import make_conformalizer
from eval.metrics import winkler_score

try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(it, total=None, disable=False):
        return it

# ---------- small utils ----------

def _product(param_grid: Dict[str, Iterable[Any]]):
    keys = list(param_grid.keys())
    grids = [list(param_grid[k]) for k in keys]
    def rec(acc, i):
        if i == len(keys):
            yield dict(acc)
        else:
            for v in grids[i]:
                acc[keys[i]] = v
                yield from rec(acc, i+1)
    if not keys:
        yield {}
    else:
        yield from rec({}, 0)

def _sample_candidates(param_grid: Dict[str, Iterable[Any]], n_trials: int, seed: Optional[int]) -> List[Dict[str,Any]]:
    all_cands = list(_product(param_grid))
    if n_trials is None or n_trials >= len(all_cands):
        return all_cands
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_cands), size=n_trials, replace=False)
    return [all_cands[i] for i in idx]

def _param_str(d: Dict[str, Any]) -> str:
    if not d: return "(none)"
    return ", ".join(f"{k}={d[k]}" for k in sorted(d.keys()))

def _safe(tag: str) -> str:
    tag = (tag or "").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")

# ---------- shared CSV appender with lock ----------

try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False

class _SharedCSVLogger:
    _DEFAULT_FIELDS = [
        "ts_iso", "array_id", "rank", "tasks",
        "method", "window_tag", "trial_idx", "total", "label",
        "val_score", "elapsed_sec", "params",
    ]
    def __init__(self, csv_path: Path, fieldnames: Optional[List[str]] = None):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames or self._DEFAULT_FIELDS

    def _lock(self, fh):
        if _HAS_FCNTL:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock(self, fh):
        if _HAS_FCNTL:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def append(self, row: Dict[str, Any], retries: int = 20, sleep_s: float = 0.05):
        payload = {k: row.get(k, "") for k in self.fieldnames}
        for _ in range(retries):
            try:
                with self.csv_path.open("a+", encoding="utf-8", newline="") as f:
                    self._lock(f)
                    if self.csv_path.stat().st_size == 0:
                        writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                        writer.writeheader()
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                    writer.writerow(payload)
                    self._unlock(f)
                return
            except OSError as e:
                if e.errno in (errno.EACCES, errno.EAGAIN):
                    time.sleep(sleep_s); continue
                raise
        raise RuntimeError(f"Failed to append to {self.csv_path} after {retries} retries.")

def _resolve_shared_csv_path(log_dir: Optional[str], method_name: str) -> Path:
    env_path = os.environ.get("TUNING_CSV")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute() and log_dir:
            p = Path(log_dir) / p
        return p
    base = f"tuning_{_safe(method_name.lower())}.csv"
    if log_dir:
        return Path(log_dir) / base
    return Path(base)

def _slurm_ctx() -> Dict[str, Any]:
    return {
        "array_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "rank": os.environ.get("SLURM_PROCID", ""),
        "tasks": os.environ.get("SLURM_NTASKS", ""),
    }

# ---------- public logger facade ----------

class TuningLogger:
    def __init__(self, enabled: bool = True, log_dir: Optional[str] = None, trial_tag: Optional[str] = None, method_name: Optional[str] = None):
        self.enabled = enabled
        self.window_tag = trial_tag or ""
        self.csv_path = _resolve_shared_csv_path(log_dir, method_name or "method")
        self.shared = _SharedCSVLogger(self.csv_path)

    def stage(self, msg: str):
        if self.enabled: print(f"[TUNE] {msg}", flush=True)

    def trial(self, idx: int, total: int, params: Dict[str, Any], score: float, elapsed: float, label: str = "val_score", method_name: str = "method"):
        if self.enabled:
            print(f"[TUNE] ({idx}/{total}) params: {_param_str(params)}  -> {label}={score:.6f}  ({elapsed:.2f}s)", flush=True)
        now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
        ctx = _slurm_ctx()
        row = {
            "ts_iso": now,
            "array_id": ctx["array_id"],
            "rank": ctx["rank"],
            "tasks": ctx["tasks"],
            "method": method_name,
            "window_tag": self.window_tag,
            "trial_idx": idx,
            "total": total,
            "label": label,
            "val_score": score,
            "elapsed_sec": elapsed,
            "params": _param_str(params),
        }
        self.shared.append(row)

# ---------- search API ----------

def search_winkler(
    method_name: str,
    param_grid: Dict[str, Iterable[Any]],
    y_train: np.ndarray, Q_train: np.ndarray, cov_train: np.ndarray | None,
    y_val: np.ndarray,   Q_val:  np.ndarray, cov_val:  np.ndarray | None,
    *,
    strategy: str = "grid",
    n_trials: Optional[int] = None,
    random_state: Optional[int] = 0,
    max_time_s: Optional[float] = None,
    tune_resource_override: Optional[Dict[str,Any]] = None,
    verbose: int = 1,
    log_dir: Optional[str] = None,
    trial_tag: Optional[str] = None,
    value_bounds: Optional[tuple[float,float]] = None,
    objective_name: str = "winkler",
) -> Tuple[Dict[str, Any], float]:

    logger = TuningLogger(
        enabled=(verbose >= 1),
        log_dir=log_dir,
        trial_tag=trial_tag,
        method_name=method_name,
    )

    if strategy not in ("grid", "random"):
        raise ValueError("strategy must be 'grid' or 'random'")
    if strategy == "grid":
        candidates = list(_product(param_grid))
    else:
        candidates = _sample_candidates(param_grid, n_trials or 32, seed=random_state)

    total = len(candidates)
    if verbose >= 1:
        logger.stage(
            f"method={method_name}  strategy={strategy}  grid_size={len(list(_product(param_grid)))}  "
            f"trials={total}  objective={objective_name}  tag={trial_tag or '-'}  csv={logger.csv_path}"
        )

    best_params, best_score = None, float("inf")
    iterator = candidates if verbose < 2 else _tqdm(candidates, total=total, disable=False)
    t_start = time.perf_counter()

    for i, params in enumerate(iterator, start=1):
        if max_time_s is not None and (time.perf_counter() - t_start) > max_time_s:
            if verbose >= 1: logger.stage(f"time budget reached ({max_time_s:.1f}s) — stopping early at {i-1} trials")
            break

        eval_params = dict(params)
        if tune_resource_override:
            eval_params.update(tune_resource_override)

        t0 = time.perf_counter()
        label = "val_score"
        try:
            conf = make_conformalizer(method_name, **{**eval_params, "value_bounds": value_bounds})
            conf.fit(y_train, Q_train, past_cov=cov_train)

            if method_name.lower() == "hopcpt" and objective_name.lower() == "hop_mse":
                if cov_val is None:
                    raise ValueError("cov_val is required for HopCPT 'hop_mse' objective.")
                score = conf.validation_recon_loss(y_val, Q_val, cov_val)
                label = "val_hop_mse"
            else:
                try:
                    pred_val = conf.batch_forecast(Q_val, future_cov=cov_val, actual_future=y_val)
                except TypeError:
                    pred_val = conf.batch_forecast(Q_val, future_cov=cov_val)
                score = np.mean(list(winkler_score(y_val, pred_val).values()))
                label = "val_winkler"

        except Exception as e:
            score = float("inf")
            if verbose >= 1:
                logger.stage(f"trial {i}/{total} failed: {_param_str(eval_params)}  err={type(e).__name__}: {e}")

        elapsed = time.perf_counter() - t0
        logger.trial(i, total, eval_params, score, elapsed, label=label, method_name=method_name)

        if score < best_score:
            best_params, best_score = params, score

    if verbose >= 1:
        logger.stage(f"best: {_param_str(best_params or {})}  val_{objective_name}={best_score:.6f}")
    return best_params or {}, best_score


def grid_search_winkler(
    method_name,
    param_grid,
    y_train, Q_train, cov_train,
    y_val,   Q_val,   cov_val,
    *,
    verbose: int = 1,
    log_dir: str | None = None,
    trial_tag: str | None = None,
    value_bounds: tuple[float,float] | None = None,
    objective_name: str = "winkler",
):
    return search_winkler(
        method_name=method_name,
        param_grid=param_grid,
        y_train=y_train, Q_train=Q_train, cov_train=cov_train,
        y_val=y_val,     Q_val=Q_val,     cov_val=cov_val,
        strategy="grid",
        n_trials=None,
        random_state=0,
        max_time_s=None,
        tune_resource_override=None,
        verbose=verbose,
        log_dir=log_dir,
        trial_tag=trial_tag,
        value_bounds=value_bounds,
        objective_name=objective_name,
    )
