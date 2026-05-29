# conformal/factory.py
from __future__ import annotations
from typing import Dict, Any

# ---- Search spaces used by tuning ----
SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    "cqr": {},
    "kmeans": {"n_clusters": [3, 5, 8, 12]},
    "knn": {"nneighbors": [50, 100, 200, 500, 1000]},
    "kernel": {
        "kernel": ["rbf", "laplacian"],
        "gamma": [0.5, 1.0, 2.0],
    },
    "nexcp": {"rho": [0.95, 0.98, 0.995]},
    "adaptive": {"gamma": [1e-4, 5e-4, 1e-3]},
    "hopcpt": {
        "hidden_dim":   [64],
        "emb_dim":      [64],
        "beta":         [5.0, 10.0, 20.0],
        "topk":         [64],
        "cosine":       [True],
        "epochs":       [20],
        "lr":           [3e-4],
        "weight_decay": [1e-5],
    },
    "nrel": {},  # baseline handled in runner
}

# Methods that require external covariates
METHODS_REQUIRING_COVS = {"kmeans", "knn", "kernel", "hopcpt"}

def requires_covariates(method_name: str) -> bool:
    return method_name.lower() in METHODS_REQUIRING_COVS


# ---- Lazy import helpers ----
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

    if key == "nrel":
        raise ValueError(
            "Method 'nrel' is a pass-through baseline and should be handled in the runner, "
            "not via make_conformalizer()."
        )

    raise ValueError(
        f"Unknown conformalizer: {name!r}. "
        "Available: cqr, kmeans, knn, kernel, nexcp, adaptive, hopcpt, nrel"
    )


__all__ = [
    "SEARCH_SPACES",
    "METHODS_REQUIRING_COVS",
    "requires_covariates",
    "make_conformalizer",
]
