#!/usr/bin/env python
import argparse
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

from dataio.loaders import load_miso_like
from runner.experiment import run_all_sites

import os, re, argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_level", required=True,
                   choices=["miso_sites", "miso_zones", "miso_system", "system_intra_1h", "system_intra_2h","spp_load_system", "wind_system" , "nyiso_system" , "spp_system" , "spp_sites" , "ercot_system",'spp_zones' , "ercot_sites" , "spp_copula" , "miso_copula" , "ercot_copula" , "spp_copula_zones","miso_copula_zones", "ercot_copula_zones","miso_zones_copula",'spp_zones_copula','wind_test','miso_load_system'])
    p.add_argument("--methods", nargs="+", required=True,
                   help="Methods to evaluate, e.g., NREL CQR KMeans KNN Kernel NexCP HopCPT Adaptive")
    p.add_argument("--covariates", nargs="+", default=None,
                   help="Superset of covariate blocks to load (we may tune over subsets).")
    p.add_argument("--test_window_days", type=int, default=1)
    p.add_argument("--out_file", type=str, default="results.csv")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--log_dir", type=str, default=None)

    # NEW: run a single site (for SLURM array task)
    p.add_argument("--site_idx", type=int, default=None,
                   help="If set, run ONLY this site index (0-based).")

    # Optional (harmless for non-HopCPT methods)
    p.add_argument("--hop_search", type=str, default="grid", choices=["grid", "random"],
                   help="Search strategy for HopCPT hyperparams (ignored by others).")
    # Control covariate selection behavior
    p.add_argument("--cov_selection", action=argparse.BooleanOptionalAction, default=True,
                   help="If True (default), tune covariate subset selection as a hyperparameter. Pass --no_cov_selection to use the full covariate set without selection.")
    # Control hyperparameter search strategy
    p.add_argument("--search_strategy", type=str, default="grid", choices=["grid", "random"],
                   help="Search strategy for hyperparameter tuning: 'grid' for exhaustive search, 'random' for random sampling.")
    p.add_argument("--n_trials", type=int, default=None,
                   help="Number of random trials when using random search strategy. If None, uses a default based on search space size.")
    p.add_argument("--cov_selection_csv", type=str, default=None,
                   help="If set, save per-window covariate selection records to this CSV path.")
    p.add_argument("--hp_csv", type=str, default=None,
                   help="If set, save per-window tuned hyperparameters and split sizes to this CSV path.")
    p.add_argument("--flavor", type=str, default="solar", choices=["solar", "wind", "load"],
                   help="Data flavor: 'solar' applies sunrise/sunset masking; 'wind'/'load' use all hours.")
    return p.parse_args()


def main():
    args = parse_args()

    need_cov = {"kmeans", "knn", "kernel", "hopcpt"}
    if any(m.lower() in need_cov for m in args.methods) and not args.covariates:
        raise SystemExit("--covariates are required for Kmeans, KNN, Kernel, or HopCPT.")

    # Load data (+ cov_blocks map if your loaders supports it)
    # If your loaders.py doesn't support return_cov_map=True, change to: actuals, marginals, cov, time_index = load_miso_like(...)
    try:
        actuals, marginals, capacities, cov, time_index, cov_blocks = load_miso_like(
            args.exp_level, args.covariates, return_cov_map=True
        )
    except TypeError:
        actuals, marginals, capacities, cov, time_index = load_miso_like(args.exp_level, args.covariates)
        cov_blocks = None

    # Decide which sites to run
    if args.site_idx is not None:
        site_indices = [args.site_idx]
        # auto-name per-site output unless user provided a path
        if args.out_file == "results.csv":
            args.out_file = f"result_{args.exp_level}_site_{args.site_idx}.csv"
    else:
        site_indices = list(range(actuals.shape[0]))

    # Run (this will average across the provided site_indices)
    df = run_all_sites(
        actuals=actuals,
        marginals=marginals,
        capacities=capacities,
        cov=cov,
        time_index=time_index,
        methods=args.methods,
        cov_blocks=cov_blocks,
        cov_names_allowed=args.covariates,
        val_days=7,
        test_window_days=args.test_window_days,
        value_bounds=(0.0, 1.0),
        verbose=args.verbose,
        log_dir=args.log_dir,
        hop_search=args.hop_search,
        exp_level=args.exp_level,
        flavor=args.flavor,
        site_indices=site_indices,       # ← key line for arrays
        cov_selection_tune=args.cov_selection,  # ← new parameter
        search_strategy=args.search_strategy,  # ← new parameter for general search strategy
        n_trials=args.n_trials,  # ← new parameter for random trials
        cov_selection_csv=args.cov_selection_csv,
        hp_csv=args.hp_csv,
    )
    df.to_csv(args.out_file, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
