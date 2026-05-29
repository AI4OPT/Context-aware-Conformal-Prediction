#!/usr/bin/env python
"""
run_experiment_csv.py

Run a CP experiment and save per-timestep prediction intervals to CSV.

Each row corresponds to one timestep in a test window. Columns:
  timestamp, site, window,
  initial_lower, initial_value, initial_upper,   # raw marginal quantiles at --level
  actual,
  <method>_lower, <method>_upper,                # CP-adjusted bounds at --level (one pair per method)

All values are in the original (denormalized) scale.

Example:
  python run_experiment_csv.py \
      --exp_level spp_system \
      --methods CQR NexCP \
      --level 0.9 \
      --out_csv intervals.csv
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from conformal.factory import make_conformalizer, SEARCH_SPACES
from dataio.loaders import load_miso_like
from eval.metrics import _li_ui_for_level
from eval.tuning import search_winkler
from runner.experiment import (
    _build_cov_matrix,
    _iter_product,
    _neighbor_vals,
    _nonempty_subsets,
    monotone_clamp,
    normalize_by_capacity,
    rolling_windows,
    split_calibration,
    vprint,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CP experiment and save per-timestep intervals to CSV."
    )
    p.add_argument("--exp_level", required=True,
                   choices=[
                       "miso_sites", "miso_zones", "miso_system", "system_intra_1h", "system_intra_2h",
                       "wind_system", "nyiso_system", "spp_system", "spp_sites",
                       "ercot_system", "spp_zones", "ercot_sites",
                       "spp_copula", "miso_copula", "ercot_copula",
                       "spp_copula_zones", "miso_copula_zones", "ercot_copula_zones",
                       "miso_zones_copula", "spp_zones_copula", "wind_test",
                   ])
    p.add_argument("--methods", nargs="+", required=True,
                   help="Methods to evaluate, e.g. NREL CQR NexCP KMeans KNN Kernel HopCPT Adaptive FEA")
    p.add_argument("--level", type=float, default=0.9,
                   help="Confidence level for interval bounds in the CSV (e.g. 0.9 → 90%% PI).")
    p.add_argument("--covariates", nargs="+", default=None,
                   help="Covariate blocks to load (required for KMeans/KNN/Kernel/HopCPT).")
    p.add_argument("--test_window_days", type=int, default=1)
    p.add_argument("--val_days", type=int, default=7)
    p.add_argument("--out_csv", type=str, default="intervals.csv",
                   help="Output CSV path.")
    p.add_argument("--site_idx", type=int, default=None,
                   help="If set, process only this site index (0-based).")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--log_dir", type=str, default=None)

    # Tuning knobs (mirrored from main.py)
    p.add_argument("--hop_search", type=str, default="grid", choices=["grid", "random"])
    p.add_argument("--cov_selection", action=argparse.BooleanOptionalAction, default=True,
                   help="If True (default), tune covariate subset selection as a hyperparameter. Pass --no_cov_selection to use the full covariate set without selection.")
    p.add_argument("--search_strategy", type=str, default="grid", choices=["grid", "random"])
    p.add_argument("--n_trials", type=int, default=None)
    p.add_argument("--flavor", type=str, default="solar", choices=["solar", "wind", "load"],
                   help="Data flavor: 'solar' applies sunrise/sunset masking; 'wind'/'load' use all hours.")
    args = p.parse_args()
    # Accept percentage input (e.g. 90) and convert to probability (0.9)
    if args.level > 1.0:
        args.level = args.level / 100.0
    if not (0.0 < args.level < 1.0):
        p.error(f"--level must be in (0, 1), got {args.level}")
    return args


# ---------------------------------------------------------------------------
# Per-site interval collection (mirrors run_one_site but records per timestep)
# ---------------------------------------------------------------------------

def collect_site_intervals(
    site_idx: int,
    y_site: np.ndarray,
    Q_site: np.ndarray,
    cap_site: float,
    cov_site: Optional[np.ndarray],
    time_index: pd.DatetimeIndex,
    methods: List[str],
    level: float,
    *,
    cov_blocks: Optional[dict] = None,
    cov_names_allowed: Optional[List[str]] = None,
    val_days: int = 7,
    test_window_days: int = 1,
    value_bounds: Tuple[float, float] = (0.0, 1.0),
    verbose: int = 1,
    log_dir: Optional[str] = None,
    hop_search: str = "grid",
    hop_random_trials: int = 48,
    hop_tune_epochs: int = 15,
    hop_refit_epochs: int = 25,
    hop_coarse_fine: bool = True,
    exp_level: str = "spp_system",
    flavor: str = "solar",  # "solar" | "wind" | "load" — controls hour masking
    cov_selection_tune: bool = True,
    search_strategy: str = "grid",
    n_trials: Optional[int] = None,
) -> List[dict]:
    """
    Return a list of row dicts — one per test-set timestep — with raw and
    CP-adjusted prediction intervals for every requested method.
    """
    vprint(f"[site={site_idx}] start", verbose)

    li, ui = _li_ui_for_level(level)   # lower / upper quantile indices for `level`
    mid_idx = 49                        # 50th percentile → point estimate

    y_norm, Q_norm, _ = normalize_by_capacity(y_site, Q_site, cap_site)
    if flavor == "solar":
        mask = y_norm > 0
    else:
        mask = np.ones(len(y_norm), dtype=bool)
    if mask.sum() < 100:
        vprint(f"[site={site_idx}] skipped (insufficient positive samples)", verbose)
        return []

    y = y_norm[mask]
    Q = Q_norm[:, mask]
    ti = time_index[mask]
    cov_site_full = cov_site[:, mask] if cov_site is not None else None

    if exp_level in ("ercot_system", "ercot_sites", "ercot_copula", "ercot_copula_zones"):
        train_start = pd.Timestamp("2018-01-01")
        first_train_end = pd.Timestamp("2018-03-01")
    else:
        train_start = pd.Timestamp("2019-01-01")
        first_train_end = pd.Timestamp("2019-03-01")

    windows = rolling_windows(ti, train_start, first_train_end, test_window_days)
    if not windows:
        vprint(f"[site={site_idx}] no windows produced", verbose)
        return []

    all_rows: List[dict] = []

    for w_idx, (cal_mask, test_mask) in enumerate(windows):
        tr_core_mask, val_mask, cal_mask = split_calibration(ti, cal_mask, val_days=val_days)

        y_trc, Q_trc   = y[tr_core_mask], Q[:, tr_core_mask]
        y_val, Q_val   = y[val_mask],     Q[:, val_mask]
        y_te,  Q_te    = y[test_mask],    Q[:, test_mask]
        y_cal_full, Q_cal_full = y[cal_mask], Q[:, cal_mask]

        timestamps_te = ti[test_mask]
        T_te = len(y_te)

        # Build per-method predictions for this window
        method_preds: Dict[str, np.ndarray] = {}  # method → (T_te, 99)

        for method in methods:
            mkey = method.lower()

            if mkey == "nrel":
                vprint(f"[site={site_idx}] w={w_idx} method={mkey}", verbose)
                pred_te = np.apply_along_axis(monotone_clamp, 0, Q_te.copy(), value_bounds).T
                method_preds[mkey] = pred_te
                continue

            space = SEARCH_SPACES.get(mkey, {})
            if mkey == "cqr":
                space = {}

            if mkey == "hopcpt":
                obj_name = "hop_mse"
                strategy = hop_search
                _n_trials = hop_random_trials if strategy == "random" else None
                T_cal = len(y_trc)
                frac_list = [0.05, 0.10, 0.20]
                topk_dyn = [None] + [max(16, min(T_cal - 1, int(fr * T_cal))) for fr in frac_list]
                topk_dyn = sorted(set(topk_dyn), key=lambda x: (x is None, x))
                if "topk" in space:
                    space = dict(space); space["topk"] = topk_dyn
                tune_override = {"epochs": hop_tune_epochs}
            else:
                obj_name = "winkler"
                strategy = search_strategy
                _n_trials = n_trials if strategy == "random" else None
                tune_override = None

            cov_flexible_methods = {"kmeans", "knn", "kernel", "hopcpt", "fea"}
            if mkey in cov_flexible_methods:
                cov_dependent = bool(cov_blocks and cov_names_allowed)
                candidate_subsets = (
                    _nonempty_subsets(cov_names_allowed)
                    if cov_selection_tune and cov_dependent
                    else ([cov_names_allowed] if cov_dependent else [None])
                )
            else:
                cov_dependent = mkey in {"kmeans", "knn", "kernel", "hopcpt"}
                candidate_subsets = (
                    _nonempty_subsets(cov_names_allowed)
                    if cov_dependent and cov_blocks and cov_names_allowed
                    else [None]
                )

            best_global_params = None
            best_global_score = float("inf")
            best_global_subset = None
            best_cov_cal = best_cov_te = best_cov_trc = best_cov_val = None

            for cov_subset in candidate_subsets:
                cov_trc = _build_cov_matrix(cov_site_full, tr_core_mask, cov_blocks, cov_subset)
                cov_val = _build_cov_matrix(cov_site_full, val_mask,     cov_blocks, cov_subset)
                cov_cal = _build_cov_matrix(cov_site_full, cal_mask,     cov_blocks, cov_subset)
                cov_teM = _build_cov_matrix(cov_site_full, test_mask,    cov_blocks, cov_subset)

                if cov_dependent and any(x is None for x in (cov_trc, cov_val, cov_cal, cov_teM)):
                    continue

                tag_cov = "-" if not cov_subset else "+".join(cov_subset)
                vprint(f"[site={site_idx}] [TUNE] w={w_idx} method={mkey} cov={tag_cov}", verbose)

                params, score = search_winkler(
                    method_name=mkey,
                    param_grid=space,
                    y_train=y_trc, Q_train=Q_trc, cov_train=cov_trc,
                    y_val=y_val,   Q_val=Q_val,   cov_val=cov_val,
                    strategy=strategy,
                    n_trials=_n_trials,
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

            # Optional HopCPT coarse→fine
            if mkey == "hopcpt" and hop_coarse_fine and best_global_params:
                master = SEARCH_SPACES.get("hopcpt", {})
                fine_space = {
                    "hidden_dim":   _neighbor_vals(best_global_params.get("hidden_dim", 64), master.get("hidden_dim", [64])),
                    "emb_dim":      _neighbor_vals(best_global_params.get("emb_dim", 64),    master.get("emb_dim", [64])),
                    "beta":         _neighbor_vals(best_global_params.get("beta", 10.0),     master.get("beta", [10.0, 20.0])),
                    "topk":         _neighbor_vals(best_global_params.get("topk", None),     space.get("topk", [None, 256])),
                    "cosine":       [best_global_params.get("cosine", True)],
                    "lr":           _neighbor_vals(best_global_params.get("lr", 1e-3),       master.get("lr", [1e-3])),
                    "weight_decay": _neighbor_vals(best_global_params.get("weight_decay", 1e-4), master.get("weight_decay", [1e-4])),
                    "epochs":       [hop_tune_epochs],
                }
                tag_cov = "-" if not best_global_subset else "+".join(best_global_subset)
                vprint(f"[site={site_idx}] [FINE] w={w_idx} method={mkey} cov={tag_cov}", verbose)
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

            if mkey == "hopcpt" and best_global_params is not None:
                best_global_params = dict(best_global_params)
                if best_global_params.get("epochs", 0) < hop_refit_epochs:
                    best_global_params["epochs"] = hop_refit_epochs

            tag_cov = "-" if not best_global_subset else "+".join(best_global_subset or [])
            vprint(f"[site={site_idx}] [REFIT] w={w_idx} method={mkey} cov={tag_cov} params={best_global_params or {}}", verbose)

            conf = make_conformalizer(mkey, **{**(best_global_params or {}), "value_bounds": value_bounds})
            conf.fit(y_cal_full, Q_cal_full, past_cov=best_cov_cal if cov_dependent else None)

            vprint(f"[site={site_idx}] [TEST ] w={w_idx} method={mkey}", verbose)
            try:
                pred_te = conf.batch_forecast(Q_te, future_cov=best_cov_te if cov_dependent else None, actual_future=y_te)
            except TypeError:
                pred_te = conf.batch_forecast(Q_te, future_cov=best_cov_te if cov_dependent else None)

            method_preds[mkey] = pred_te  # (T_te, 99)

        # Build rows for this window (all values normalized by capacity)
        for t in range(T_te):
            row = {
                "timestamp":     timestamps_te[t],
                "site":          site_idx,
                "window":        w_idx,
                "actual":        float(y_te[t]),
                "initial_lower": float(Q_te[li, t]),
                "initial_value": float(Q_te[mid_idx, t]),
                "initial_upper": float(Q_te[ui, t]),
            }
            for mkey, pred in method_preds.items():
                row[f"{mkey}_lower"] = float(pred[t, li])
                row[f"{mkey}_upper"] = float(pred[t, ui])
            all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    need_cov = {"kmeans", "knn", "kernel", "hopcpt"}
    if any(m.lower() in need_cov for m in args.methods) and not args.covariates:
        raise SystemExit("--covariates are required for KMeans, KNN, Kernel, or HopCPT.")

    try:
        actuals, marginals, capacities, cov, time_index, cov_blocks = load_miso_like(
            args.exp_level, args.covariates, return_cov_map=True
        )
    except TypeError:
        actuals, marginals, capacities, cov, time_index = load_miso_like(
            args.exp_level, args.covariates
        )
        cov_blocks = None

    # Ensure marginals shape = (n_sites, 99, T)
    if marginals.ndim == 3 and marginals.shape[1] != 99 and marginals.shape[2] == 99:
        marginals = np.transpose(marginals, (0, 2, 1))

    N = actuals.shape[0]
    site_indices = [args.site_idx] if args.site_idx is not None else list(range(N))

    all_rows = []
    for i in site_indices:
        rows = collect_site_intervals(
            site_idx=i,
            y_site=actuals[i],
            Q_site=marginals[i],
            cap_site=capacities[i],
            cov_site=cov[i] if cov is not None else None,
            time_index=time_index,
            methods=args.methods,
            level=args.level,
            cov_blocks=cov_blocks,
            cov_names_allowed=args.covariates,
            val_days=args.val_days,
            test_window_days=args.test_window_days,
            value_bounds=(0.0, 1.0),
            verbose=args.verbose,
            log_dir=args.log_dir,
            hop_search=args.hop_search,
            exp_level=args.exp_level,
            flavor=args.flavor,
            cov_selection_tune=args.cov_selection,
            search_strategy=args.search_strategy,
            n_trials=args.n_trials,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No data collected — check site indices and data availability.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows → {args.out_csv}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
