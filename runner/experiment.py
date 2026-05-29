from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from itertools import combinations

from conformal.factory import make_conformalizer, SEARCH_SPACES
from eval.tuning import search_winkler
from eval.metrics import picp_aiw, winkler_score


# -------------------------- utilities --------------------------

def vprint(msg: str, verbose: int):
    if verbose >= 1:
        print(msg, flush=True)

def monotone_clamp(q: np.ndarray, bounds: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Enforce non-decreasing quantiles across the 99 grid and clamp to bounds.
    q: shape (99,) or (99, T). Works column-wise if 2D.
    """
    lo, hi = bounds
    if q.ndim == 1:
        out = np.clip(q, lo, hi)
        out = np.maximum.accumulate(out)
        return out
    elif q.ndim == 2:
        out = np.clip(q, lo, hi)
        out = np.maximum.accumulate(out, axis=0)
        return out
    else:
        raise ValueError("Expected q with shape (99,) or (99, T)")

def normalize_by_capacity(actual: np.ndarray, Q: np.ndarray, cap: float) -> Tuple[np.ndarray, np.ndarray, float]:
    # cap = float(np.max(actual)) if np.max(actual) > 0 else 1.0
    y = actual / cap
    Qn = Q / cap
    return y, Qn, cap

def rolling_windows(
    time_index: pd.DatetimeIndex,
    train_start: pd.Timestamp,
    first_train_end: pd.Timestamp,
    test_window_days: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produce (cal_mask, test_mask) pairs for rolling evaluation.
    Calibration (cal) covers [train_start, train_end); Test covers [train_end, train_end + test_window).
    """
    out = []
    train_end = first_train_end
    test_off = pd.DateOffset(days=test_window_days)

    while True:
        cal_mask  = (time_index >= train_start) & (time_index < train_end)
        test_mask = (time_index >= train_end)   & (time_index < train_end + test_off)
        if not np.any(test_mask):
            break
        out.append((cal_mask, test_mask))

        train_end += test_off
        if train_end >= time_index[-1]:
            break
    return out

def split_calibration(
    time_index: pd.DatetimeIndex,
    cal_mask: np.ndarray,
    val_days: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split calibration set (all data before test) into:
      - train_core: everything in cal except the last `val_days`
      - val: the last `val_days` days inside cal
      - cal: unchanged (alias of input cal_mask)
    """
    if not np.any(cal_mask):
        empty = cal_mask & False
        return empty, empty, cal_mask

    ti_cal = time_index[cal_mask]
    cal_end = ti_cal[-1]
    val_start = cal_end - pd.DateOffset(days=val_days) + pd.Timedelta(seconds=1)

    val_mask_full = (time_index >= val_start) & (time_index <= cal_end)
    val_mask = cal_mask & val_mask_full
    train_core_mask = cal_mask & (~val_mask)
    return train_core_mask, val_mask, cal_mask

def _fmt_span(ti: pd.DatetimeIndex, mask: np.ndarray) -> str:
    if not np.any(mask):
        return "empty"
    sub = ti[mask]
    return f"{sub[0]:%Y-%m-%d %H:%M} → {sub[-1]:%Y-%m-%d %H:%M} (T={sub.size})"

def _iter_product(param_grid: Dict[str, List]) -> List[Dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    from itertools import product
    for combo in product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, combo))

def _neighbor_vals(val, choices):
    arr = sorted(choices, key=lambda x: (x is None, x))
    if val not in arr:
        if isinstance(val, (int, float)):
            idx = int(np.argmin([abs((c if c is not None else -1e9) - val) for c in arr]))
        else:
            idx = 0
    else:
        idx = arr.index(val)
    out = {arr[max(0, idx - 1)], arr[idx], arr[min(len(arr) - 1, idx + 1)]}
    return list(sorted(out, key=lambda x: (x is None, x)))

def _nonempty_subsets(names: List[str]) -> List[List[str]]:
    out = []
    for r in range(1, len(names) + 1):
        out.extend([list(x) for x in combinations(names, r)])
    return out

def _build_cov_matrix(
    cov_site: Optional[np.ndarray],  # (K_total, T) for this site (already normalized)
    ti_mask: np.ndarray,             # boolean mask over time T
    cov_blocks: Optional[dict[str, slice]],
    subset: Optional[List[str]],
) -> Optional[np.ndarray]:
    if cov_site is None or not subset or not cov_blocks:
        return None
    blocks = []
    for name in subset:
        sl = cov_blocks.get(name)
        if sl is None:
            continue
        blocks.append(cov_site[sl, :])  # (k_c, T)
    if not blocks:
        return None
    M = np.concatenate(blocks, axis=0)  # (sum_k, T)
    return M[:, ti_mask].T               # -> (T_subset, sum_k)


# -------------------------- metrics aggregation --------------------------

LEVELS = [0.9, 0.8, 0.7, 0.6]

@dataclass
class MethodMetrics:
    picp: Dict[float, List[float]]
    aiw: Dict[float, List[float]]
    winkler: Dict[float, List[float]]

    @classmethod
    def empty(cls):
        return cls(
            picp={c: [] for c in LEVELS},
            aiw={c: [] for c in LEVELS},
            winkler={c: [] for c in LEVELS},
        )

    def update_from_preds(self, y_true: np.ndarray, pred_Q: np.ndarray):
        picp_d, aiw_d = picp_aiw(y_true, pred_Q)
        wink_d = winkler_score(y_true, pred_Q)
        for c in LEVELS:
            self.picp[c].append(picp_d[c])
            self.aiw[c].append(aiw_d[c])
            self.winkler[c].append(wink_d[c])

    def summarize(self) -> Dict[str, Dict[float, float]]:
        return {
            "picp": {c: float(np.mean(self.picp[c])) if self.picp[c] else np.nan for c in LEVELS},
            "aiw": {c: float(np.mean(self.aiw[c])) if self.aiw[c] else np.nan for c in LEVELS},
            "winkler": {c: float(np.mean(self.winkler[c])) if self.winkler[c] else np.nan for c in LEVELS},
        }


# -------------------------- per-site runner --------------------------

def run_one_site(
    site_idx: int,
    y_site: np.ndarray,
    Q_site: np.ndarray,
    cap_site: float,
    cov_site: Optional[np.ndarray],
    time_index: pd.DatetimeIndex,
    methods: List[str],
    *,
    cov_blocks: Optional[dict[str, slice]] = None,
    cov_names_allowed: Optional[List[str]] = None,
    val_days: int = 7,
    test_window_days: int = 7,
    value_bounds: Tuple[float, float] = (0.0, 1.0),
    verbose: int = 1,
    log_dir: Optional[str] = None,
    # HopCPT tuning knobs
    hop_random_trials: int = 48,
    hop_tune_epochs: int = 15,
    hop_refit_epochs: int = 25,
    hop_coarse_fine: bool = True,
    # >>> ADD THIS <<<
    hop_search: str = "random",
    exp_level: str = "spp_system",  # "spp_system" or "ercot_system"
    # "grid" or "random" (ignored for non-HopCPT)
    flavor: str = "solar",  # "solar" | "wind" | "load" — controls hour masking
    cov_selection_tune: bool = True,  # whether to tune covariate selection as hyperparameter
    search_strategy: str = "grid",  # search strategy for hyperparameter tuning
    n_trials: Optional[int] = None,  # number of trials for random search
) -> Tuple[Dict[str, MethodMetrics], List[Dict], List[Dict]]:
    vprint(f"[site={site_idx}] start", verbose)

    cov_records: List[Dict] = []
    hp_records: List[Dict] = []

    # Normalize & mask zeros (solar only — wind/load use all hours)
    y_norm, Q_norm, _ = normalize_by_capacity(y_site, Q_site, cap_site)
    if flavor == "solar":
        mask = y_norm > 0
    else:
        mask = np.ones(len(y_norm), dtype=bool)
    if mask.sum() < 100:
        vprint(f"[site={site_idx}] skipped (insufficient positive samples)", verbose)
        return {}, [], []

    y = y_norm[mask]
    Q = Q_norm[:, mask]
    ti = time_index[mask]

    cov_site_full = cov_site[:, mask] if cov_site is not None else None  # (K_total, T')

    # rolling calibration/test windows
    if exp_level == "ercot_system" or exp_level == "ercot_sites" or exp_level == "ercot_copula" or exp_level == "ercot_copula_zones":
        train_start = pd.Timestamp("2018-01-01")
        first_train_end = pd.Timestamp("2018-03-01")
    else:
        train_start = pd.Timestamp("2019-01-01")
        first_train_end = pd.Timestamp("2019-03-01")
    windows = rolling_windows(ti, train_start, first_train_end, test_window_days)
    if not windows:
        vprint(f"[site={site_idx}] no windows produced", verbose)
        return {}, [], []

    metrics_by_method: Dict[str, MethodMetrics] = {m.lower(): MethodMetrics.empty() for m in methods}

    for w_idx, (cal_mask, test_mask) in enumerate(windows):
        # split calibration into train_core / val
        tr_core_mask, val_mask, cal_mask = split_calibration(ti, cal_mask, val_days=val_days)

        # log spans
        if verbose >= 1:
            print(
                f"[site={site_idx}] window={w_idx}\n"
                f"  train_core: {_fmt_span(ti, tr_core_mask)}\n"
                f"  val      : {_fmt_span(ti, val_mask)}\n"
                f"  cal(full): {_fmt_span(ti, cal_mask)}\n"
                f"  test     : {_fmt_span(ti, test_mask)}",
                flush=True
            )

        # slices
        y_trc, Q_trc = y[tr_core_mask], Q[:, tr_core_mask]
        y_val,  Q_val  = y[val_mask],  Q[:, val_mask]
        y_te,   Q_te   = y[test_mask], Q[:, test_mask]
        y_cal_full, Q_cal_full = y[cal_mask], Q[:, cal_mask]

        for method in methods:
            mkey = method.lower()

            # NREL pass-through baseline
            if mkey == "nrel":
                vprint(f"[site={site_idx}] [STAGE=TEST ] method={mkey}", verbose)
                pred_te = np.apply_along_axis(monotone_clamp, 0, Q_te.copy(), value_bounds).T
                metrics_by_method[mkey].update_from_preds(y_te, pred_te)
                hp_records.append({
                    "site": site_idx, "window": w_idx, "method": mkey,
                    "test_start": str(ti[test_mask][0]) if np.any(test_mask) else "",
                    "test_end":   str(ti[test_mask][-1]) if np.any(test_mask) else "",
                    "n_cal": int(cal_mask.sum()), "n_test": int(test_mask.sum()),
                })
                continue

            space = SEARCH_SPACES.get(mkey, {})
            if mkey == "cqr":
                space = {}

            if mkey == "hopcpt":
                obj_name = "hop_mse"
                strategy = hop_search  # Use hop_search parameter for HopCPT
                n_trials = hop_random_trials if strategy == "random" else None
                # dynamic topk choices based on calibration size
                T_cal = len(y_trc)
                frac_list = [0.05, 0.10, 0.20]
                topk_dyn = [None] + [max(16, min(T_cal - 1, int(fr * T_cal))) for fr in frac_list]
                topk_dyn = sorted(set(topk_dyn), key=lambda x: (x is None, x))
                if "topk" in space:
                    space = dict(space); space["topk"] = topk_dyn
                tune_override = {"epochs": hop_tune_epochs}
            else:
                obj_name = "winkler"
                strategy = search_strategy  # Use configurable search strategy for other methods
                n_trials = n_trials if strategy == "random" else None
                tune_override = None

            # Methods that can work with or without covariates
            cov_flexible_methods = {"kmeans", "knn", "kernel", "hopcpt", "fea"}

            # Determine if this method should use covariate selection
            if mkey in cov_flexible_methods:
                # For flexible methods, use covariate selection if available
                cov_dependent = cov_blocks and cov_names_allowed
                if cov_selection_tune and cov_dependent:
                    # When cov_selection_tune is enabled, try different covariate subsets
                    candidate_subsets = _nonempty_subsets(cov_names_allowed)
                else:
                    # Otherwise, use all available covariates or none
                    candidate_subsets = [cov_names_allowed] if cov_dependent else [None]
            else:
                # Original logic: only these methods are strictly cov-dependent
                cov_dependent = mkey in {"kmeans", "knn", "kernel", "hopcpt"}
                candidate_subsets = _nonempty_subsets(cov_names_allowed) if cov_dependent and cov_blocks and cov_names_allowed else [None]

            best_global_params = None
            best_global_score = float("inf")
            best_global_subset: Optional[List[str]] = None

            # try all cov subsets (for cov-dependent methods)
            for cov_subset in candidate_subsets:
                # Build cov matrices for this subset
                cov_trc = _build_cov_matrix(cov_site_full, tr_core_mask, cov_blocks, cov_subset)
                cov_val = _build_cov_matrix(cov_site_full, val_mask,     cov_blocks, cov_subset)
                cov_cal = _build_cov_matrix(cov_site_full, cal_mask,     cov_blocks, cov_subset)
                cov_teM = _build_cov_matrix(cov_site_full, test_mask,    cov_blocks, cov_subset)

                if cov_dependent and (cov_trc is None or cov_val is None or cov_cal is None or cov_teM is None):
                    continue

                tag_cov = "-" if not cov_subset else "+".join(cov_subset)
                if strategy == "random":
                    vprint(f"[site={site_idx}] [STAGE=TUNE] method={mkey} cov={tag_cov} strategy={strategy} trials={n_trials}", verbose)
                else:
                    vprint(f"[site={site_idx}] [STAGE=TUNE] method={mkey} cov={tag_cov} strategy={strategy} grid={len(list(_iter_product(space)))}", verbose)

                params, score = search_winkler(
                    method_name=mkey,
                    param_grid=space,
                    y_train=y_trc, Q_train=Q_trc, cov_train=cov_trc,
                    y_val=y_val,   Q_val=Q_val,   cov_val=cov_val,
                    strategy=strategy,
                    n_trials=n_trials,
                    random_state=0,
                    max_time_s=None,
                    tune_resource_override=tune_override,
                    objective_name=obj_name,
                    value_bounds=value_bounds,
                    verbose=verbose,
                    log_dir=log_dir,
                    trial_tag=f"site{site_idx}_w{w_idx}_{mkey}_{tag_cov}",
                )

                if score < best_global_score:
                    best_global_score  = score
                    best_global_params = params
                    best_global_subset = cov_subset
                    best_cov_cal = cov_cal
                    best_cov_te  = cov_teM
                    best_cov_trc = cov_trc
                    best_cov_val = cov_val

            # record covariate selection for this (site, window, method)
            if cov_names_allowed:
                test_start = str(ti[test_mask][0]) if np.any(test_mask) else ""
                test_end   = str(ti[test_mask][-1]) if np.any(test_mask) else ""
                record: Dict = {
                    "site": site_idx,
                    "window": w_idx,
                    "method": mkey,
                    "test_start": test_start,
                    "test_end": test_end,
                    "best_score": best_global_score,
                    "best_covariates": "+".join(best_global_subset) if best_global_subset else "",
                }
                for _c in cov_names_allowed:
                    record[f"cov_{_c}"] = 1 if (best_global_subset and _c in best_global_subset) else 0
                cov_records.append(record)

            # optional coarse→fine for HopCPT (same cov subset)
            if mkey == "hopcpt" and hop_coarse_fine and best_global_params:
                master = SEARCH_SPACES.get("hopcpt", {})
                fine_space = {
                    "hidden_dim":   _neighbor_vals(best_global_params.get("hidden_dim", 64),     master.get("hidden_dim", [64])),
                    "emb_dim":      _neighbor_vals(best_global_params.get("emb_dim", 64),        master.get("emb_dim", [64])),
                    "beta":         _neighbor_vals(best_global_params.get("beta", 10.0),         master.get("beta", [10.0, 20.0])),
                    "topk":         _neighbor_vals(best_global_params.get("topk", None),         space.get("topk", [None, 256])),
                    "cosine":       [best_global_params.get("cosine", True)],
                    "lr":           _neighbor_vals(best_global_params.get("lr", 1e-3),           master.get("lr", [1e-3])),
                    "weight_decay": _neighbor_vals(best_global_params.get("weight_decay", 1e-4), master.get("weight_decay", [1e-4])),
                    "epochs":       [hop_tune_epochs],
                }
                tag_cov = "-" if not best_global_subset else "+".join(best_global_subset)
                vprint(f"[site={site_idx}] [STAGE=FINE ] method={mkey} cov={tag_cov} grid={len(list(_iter_product(fine_space)))}", verbose)
                params_f, score_f = search_winkler(
                    method_name=mkey,
                    param_grid=fine_space,
                    y_train=y_trc, Q_train=Q_trc, cov_train=best_cov_trc,
                    y_val=y_val,   Q_val=Q_val,   cov_val=best_cov_val,
                    strategy="grid",
                    n_trials=None,
                    random_state=0,
                    max_time_s=None,
                    tune_resource_override=None,
                    objective_name="hop_mse",
                    value_bounds=value_bounds,
                    verbose=verbose,
                    log_dir=log_dir,
                    trial_tag=f"site{site_idx}_w{w_idx}_{mkey}_{tag_cov}_fine",
                )
                if score_f < best_global_score:
                    best_global_params = params_f
                    best_global_score  = score_f

            # bump epochs for final refit if HopCPT
            if mkey == "hopcpt" and best_global_params is not None:
                best_global_params = dict(best_global_params)
                if best_global_params.get("epochs", 0) < hop_refit_epochs:
                    best_global_params["epochs"] = hop_refit_epochs

            # record tuned hyperparameters and split sizes for this (site, window, method)
            _hp_rec = {
                "site": site_idx, "window": w_idx, "method": mkey,
                "test_start": str(ti[test_mask][0]) if np.any(test_mask) else "",
                "test_end":   str(ti[test_mask][-1]) if np.any(test_mask) else "",
                "n_cal": int(cal_mask.sum()), "n_test": int(test_mask.sum()),
            }
            if best_global_params:
                _hp_rec.update(best_global_params)
            hp_records.append(_hp_rec)

            tag_cov = "-" if not best_global_subset else "+".join(best_global_subset or [])
            vprint(f"[site={site_idx}] [STAGE=REFIT] method={mkey} cov={tag_cov} params={best_global_params or {}}", verbose)

            conf = make_conformalizer(mkey, **{**(best_global_params or {}), "value_bounds": value_bounds})
            
            conf.fit(y_cal_full, Q_cal_full, past_cov=best_cov_cal if cov_dependent else None)

            vprint(f"[site={site_idx}] [STAGE=TEST ] method={mkey} cov={tag_cov}", verbose)
            try:
                pred_te = conf.batch_forecast(Q_te, future_cov=best_cov_te if cov_dependent else None, actual_future=y_te)
            except TypeError:
                pred_te = conf.batch_forecast(Q_te, future_cov=best_cov_te if cov_dependent else None)
            metrics_by_method[mkey].update_from_preds(y_te, pred_te)

    return metrics_by_method, cov_records, hp_records


# -------------------------- aggregate across sites --------------------------

def aggregate_sites(
    all_site_summaries: List[Dict[str, MethodMetrics]],
    methods: List[str],
) -> pd.DataFrame:
    acc = {
        m.lower(): {
            "picp": {c: [] for c in LEVELS},
            "aiw": {c: [] for c in LEVELS},
            "winkler": {c: [] for c in LEVELS},
        } for m in methods
    }

    for site_dict in all_site_summaries:
        for mkey, mm in site_dict.items():
            summ = mm.summarize()
            for c in LEVELS:
                acc[mkey]["picp"][c].append(summ["picp"][c])
                acc[mkey]["aiw"][c].append(summ["aiw"][c])
                acc[mkey]["winkler"][c].append(summ["winkler"][c])

    rows = []
    for c in LEVELS:
        row = {"Level": c}
        for method in methods:
            mkey = method.lower()
            row[f"Avg PICP ({method})"] = float(np.nanmean(acc[mkey]["picp"][c])) if acc[mkey]["picp"][c] else np.nan
            row[f"Avg AIW  ({method})"] = float(np.nanmean(acc[mkey]["aiw"][c])) if acc[mkey]["aiw"][c] else np.nan
            row[f"Avg Winkler ({method})"] = float(np.nanmean(acc[mkey]["winkler"][c])) if acc[mkey]["winkler"][c] else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------- top-level experiment --------------------------

def run_all_sites(
    actuals: np.ndarray,
    marginals: np.ndarray,
    capacities: np.ndarray,
    cov: Optional[np.ndarray],
    time_index: pd.DatetimeIndex,
    methods: List[str],
    *,
    cov_blocks: Optional[dict] = None,
    cov_names_allowed: Optional[List[str]] = None,
    val_days: int = 7,
    test_window_days: int = 7,
    value_bounds: Tuple[float, float] = (0.0, 1.0),
    verbose: int = 1,
    log_dir: Optional[str] = None,
    hop_search: str = "grid",
    hop_random_trials: int = 48,
    hop_tune_epochs: int = 15,
    hop_refit_epochs: int = 25,
    hop_coarse_fine: bool = True,
    # NEW: allow selecting which sites to evaluate (for SLURM arrays)
    site_indices: Optional[List[int]] = None,
    exp_level: str = "miso",  # "spp_system" or "ercot_system"
    flavor: str = "solar",  # "solar" | "wind" | "load" — controls hour masking
    cov_selection_tune: bool = True,  # whether to tune covariate selection as hyperparameter
    search_strategy: str = "grid",  # search strategy for hyperparameter tuning
    n_trials: Optional[int] = None,  # number of trials for random search
    cov_selection_csv: Optional[str] = None,  # path to save per-window covariate selection CSV
    hp_csv: Optional[str] = None,  # path to save per-window hyperparameter records CSV
) -> pd.DataFrame:
    """
    Evaluate methods across sites and average metrics.
    If site_indices is provided, only those sites are processed.
    """
    # Ensure marginals shape = (n, 99, t)
    if marginals.ndim == 3 and marginals.shape[1] != 99 and marginals.shape[2] == 99:
        marginals = np.transpose(marginals, (0, 2, 1))

    N = actuals.shape[0]
    if site_indices is None:
        site_indices = list(range(N))
    else:
        # basic sanity checks
        site_indices = [int(i) for i in site_indices if 0 <= int(i) < N]
        if not site_indices:
            raise ValueError("site_indices is empty after validation.")

    all_site_summaries: List[Dict[str, "MethodMetrics"]] = []
    all_cov_records: List[Dict] = []
    all_hp_records: List[Dict] = []

    for i in site_indices:
        if verbose >= 1:
            print(f"[site={i}] start")

        y_site = actuals[i]
        Q_site = marginals[i]
        cov_site = cov[i] if cov is not None else None
        cap_site = capacities[i]

        site_metrics, site_cov_records, site_hp_records = run_one_site(
            site_idx=i,
            y_site=y_site,
            Q_site=Q_site,
            cap_site=cap_site,
            cov_site=cov_site,
            cov_selection_tune=cov_selection_tune,
            search_strategy=search_strategy,
            n_trials=n_trials,
            time_index=time_index,
            methods=methods,
            cov_blocks=cov_blocks,
            cov_names_allowed=cov_names_allowed,
            val_days=val_days,
            test_window_days=test_window_days,
            value_bounds=value_bounds,
            verbose=verbose,
            log_dir=log_dir,
            hop_random_trials=hop_random_trials,
            hop_tune_epochs=hop_tune_epochs,
            hop_refit_epochs=hop_refit_epochs,
            hop_coarse_fine=hop_coarse_fine,
            hop_search=hop_search,   # harmless for non-HopCPT methods
            exp_level=exp_level,
            flavor=flavor,
        )
        if site_metrics:
            all_site_summaries.append(site_metrics)
        all_cov_records.extend(site_cov_records)
        all_hp_records.extend(site_hp_records)

    if cov_selection_csv and all_cov_records:
        pd.DataFrame(all_cov_records).to_csv(cov_selection_csv, index=False)
        if verbose >= 1:
            print(f"[cov_selection] saved {len(all_cov_records)} records → {cov_selection_csv}")

    if hp_csv and all_hp_records:
        pd.DataFrame(all_hp_records).to_csv(hp_csv, index=False)
        if verbose >= 1:
            print(f"[hp_records] saved {len(all_hp_records)} records → {hp_csv}")

    # Average metrics across all processed sites
    return aggregate_sites(all_site_summaries, methods)
