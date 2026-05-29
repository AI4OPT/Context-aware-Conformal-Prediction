import numpy as np
from typing import Dict, Tuple

def _li_ui_for_level(level: float) -> Tuple[int, int]:
    c = round(level, 2)
    lb = (1 - c) / 2
    ub = 1 - lb
    li = int(np.ceil(lb * 99)) - 1
    ui = int(np.floor(ub * 99))
    return li, ui

def picp_aiw(y: np.ndarray, pred: np.ndarray) -> (Dict[float,float], Dict[float,float]):
    # pred: (T, Q)
    levels = [0.9, 0.8, 0.7, 0.6]
    picp, aiw = {}, {}
    for c in levels:
        li, ui = _li_ui_for_level(c)
        L, U = pred[:, li], pred[:, ui]
        cover = (y >= L) & (y <= U)
        picp[c] = float(cover.mean())
        aiw[c]  = float(np.mean(U - L))
    return picp, aiw

def winkler_score(y: np.ndarray, pred: np.ndarray) -> Dict[float,float]:
    levels = [0.9, 0.8, 0.7, 0.6]
    out = {}
    for c in levels:
        li, ui = _li_ui_for_level(c)
        L, U = pred[:, li], pred[:, ui]
        alpha = 1 - c
        width = U - L
        penalty = 2/alpha * ((L - y) * (y < L) + (y - U) * (y > U))
        out[c] = float(np.mean(width + penalty))
    return out
