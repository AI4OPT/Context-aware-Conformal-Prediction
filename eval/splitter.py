import numpy as np
import pandas as pd
from typing import Iterator, Tuple

def rolling_windows(time_index: pd.DatetimeIndex,
                    train_end: pd.Timestamp,
                    test_window: pd.DateOffset) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    start = time_index[0]
    end   = time_index[-1]
    t_end = train_end
    while t_end + test_window <= end:
        tr = (time_index >= start) & (time_index < t_end)
        te = (time_index >= t_end) & (time_index < t_end + test_window)
        yield tr, te
        t_end += test_window

def split_train_val(time_index: pd.DatetimeIndex,
                    train_mask: np.ndarray,
                    val_days: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Carve the last `val_days` of the train period as validation."""
    if not train_mask.any():
        return train_mask, np.zeros_like(train_mask, dtype=bool)
    tr_idx = np.where(train_mask)[0]
    cutoff = time_index[tr_idx[-1]] - pd.Timedelta(days=val_days) + pd.Timedelta(hours=0)
    val_mask = (time_index >= cutoff) & (time_index <= time_index[tr_idx[-1]])
    val_mask = val_mask & train_mask
    core_train = train_mask & (~val_mask)
    return core_train, val_mask
