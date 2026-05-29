# conformal/factory.py
from __future__ import annotations
from typing import Dict, Any

# ---- Search spaces used by tuning ----
SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    "cqr": {},
    "kmeans": {"n_clusters": [3, 5, 8, 12]},
    # "kmeans": {"n_clusters": [3]},
    "knn": {"nneighbors": [50, 100, 200, 500, 1000]},
    # "knn": {"nneighbors": [500]},
    "kernel": {
        # "kernel": ["rbf", "laplacian", "linear", "poly"],
        "kernel": ["rbf", "laplacian"],
        "gamma": [0.5, 1.0, 2.0],
        # "degree": [2, 3],
        # "coef0": [0.0, 1.0],
    },
    "nexcp": {"rho": [0.95, 0.98, 0.995]},
    # "nexcp": {"rho": [0.999]},
    "adaptive": {"gamma": [1e-4, 5e-4, 1e-3]},
    "hopcpt": {
        "hidden_dim":   [64],
        "emb_dim":      [64],
        "beta":         [5.0, 10.0,20.0],
        "topk":         [64],
        "cosine":       [True],
        "epochs":       [20],
        "lr":           [3e-4],
        "weight_decay": [1e-5],
    },
    # NEW: FFT-derived periodic covariates + backends
    "periodic": {
        "backend":        ["kmeans"],
        # "n_periods":      [2, 3, 4],
        "min_prominence": [0.001],
        # "sampling_rate":  [1.0],
        # # kmeans:
        "n_clusters":     [3, 5, 8 ,10 ,12],
        # knn:
        # "nneighbors":     [50, 100, 200, 500, 1000],
        # "distance_weighted": [True],
        # # kernel:
        # "kernel":         ["rbf", "laplacian"],
        # "gamma":          [0.5, 1.0, 2.0],
        # # generic:
        # "scale_features": [True],
    },
    "nrel": {},  # baseline handled in runner
    "fea": {
        "hidden_dim": [32,64],  # Fixed: was "hidden", now matches constructor parameter
        "emb_dim": [32,64],
        "fea_embed_dim": [32,64],
        "fea_num_heads": [1,8,16],
        "fea_modes": [16, 32, 64],  # Fixed: was "fea_M", now matches constructor parameter
        "fea_mode_policy": ["lowfreq"],
        "fea_activation": ["softmax"],
        "fea_loss": ["winkler"],  # New: loss function option
        "fea_entropy_reg": [0.0],
        "lr": [1e-3,3e-4],
        "n_epochs": [50,100],
        "weight_decay": [0.0,1e-3],
    },
}

# Methods that require external covariates
METHODS_REQUIRING_COVS = {"kmeans", "knn", "kernel", "hopcpt", "fea"}
# NOTE: 'periodic' builds its own covariates internally from conformity periodicities,
# so it is NOT listed above.

def requires_covariates(method_name: str) -> bool:
    return method_name.lower() in METHODS_REQUIRING_COVS


# ---- Robust import helpers (lazy) ----
def _import_cqr():
    from .cqr import CQRConformalizer
    return CQRConformalizer

def _import_kmeans():
    from .kmeans import KMeansConformalizer
    return KMeansConformalizer

def _import_knn():
    from .knn import KNNConformalizer
    return KNNConformalizer

def _import_kernel():
    from .kernel import KernelConformalizer
    return KernelConformalizer

def _import_adaptive():
    from .adaptive import AdaptiveConformalizer
    return AdaptiveConformalizer

def _import_nexcp():
    from . import nexcp as _m
    for name in ("NExCPConformalizer", "NexCP", "NexcpConformalizer", "Nexcp"):
        if hasattr(_m, name):
            return getattr(_m, name)
    raise ImportError(
        "Could not find a NexCP conformalizer class in conformal/nexcp.py. "
        "Expected one of: NExCPConformalizer, NexCP, NexcpConformalizer, Nexcp."
    )

def _import_hopcpt():
    from .hopcpt_learned import HopCPTLearnedConformalizer
    return HopCPTLearnedConformalizer

def _import_periodic():
    from .periodic import PeriodicCovariateConformalizer
    return PeriodicCovariateConformalizer

def _import_fea():
    from .feacpt import FEACPTConformalizer
    return FEACPTConformalizer



# ---- Factory ----
def make_conformalizer(name: str, **kw: Any):
    """
    Create a conformalizer by name.
    NOTE: 'nrel' is a pass-through baseline and is not constructed here.
    """
    key = name.lower()

    if key == "cqr":
        return _import_cqr()(**kw)
    if key == "kmeans":
        return _import_kmeans()(**kw)
    if key == "knn":
        return _import_knn()(**kw)
    if key == "kernel":
        return _import_kernel()(**kw)
    if key == "nexcp":
        return _import_nexcp()(**kw)
    if key == "adaptive":
        return _import_adaptive()(**kw)
    if key == "hopcpt":
        return _import_hopcpt()(**kw)
    if key in ("periodic", "fftperiodic", "seasonal"):
        return _import_periodic()(**kw)

    if key == "nrel":
        raise ValueError(
            "Method 'NREL' is a pass-through baseline and should be handled in the runner, "
            "not via make_conformalizer()."
        )

    if key == "fea":
        return _import_fea()(**kw)

    raise ValueError(f"Unknown conformalizer: {name!r}. Available: cqr, kmeans, knn, kernel, nexcp, adaptive, hopcpt, periodic, nrel, fea")


__all__ = [
    "SEARCH_SPACES",
    "METHODS_REQUIRING_COVS",
    "requires_covariates",
    "make_conformalizer",
]
