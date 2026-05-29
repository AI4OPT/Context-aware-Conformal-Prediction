import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, List, Dict, Tuple


def minmax_scale_3d(x: np.ndarray) -> np.ndarray:
    """
    MinMax-scale a 3D array per feature across (n, t).

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n, k, t)

    Returns
    -------
    np.ndarray
        Scaled array of shape (n, k, t)
    """
    # x: (n, k, t) -> scale per feature over (n,t)
    n, k, t = x.shape
    reshaped = x.transpose(1, 0, 2).reshape(k, -1).T  # (n*t, k)
    scaled = MinMaxScaler().fit_transform(reshaped)    # (n*t, k)
    return scaled.T.reshape(k, n, t).transpose(1, 0, 2)  # (n, k, t)


def load_miso_like(
    exp_level: str,
    covariates: Optional[List[str]],
    return_cov_map: bool = False,
):
    base_dir = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/MISO/solar/'
    cov_base = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/cov'

    if exp_level == "miso_sites":
        actuals = np.load(f'{base_dir}day-ahead/sites/actuals.npy')
        marg = np.load(f'{base_dir}day-ahead/sites/marginals.npy')
        cap = np.load(f'{base_dir}day-ahead/sites/capacity.npy')
        cov_prefix, freq, end = "miso_sites", "1h", '2019-12-31 23:00'
    elif exp_level == "miso_zones":
        actuals = np.load(f'{base_dir}day-ahead/zones/actuals.npy')
        marg = np.load(f'{base_dir}day-ahead/zones/marginals.npy')
        cap = np.load(f'{base_dir}day-ahead/zones/capacity.npy')
        cov_prefix, freq, end = "miso_zones", "1h", '2019-12-31 23:00'
    elif exp_level == "miso_system":
        actuals = np.load(f'{base_dir}day-ahead/system/actuals.npy')
        marg = np.load(f'{base_dir}day-ahead/system/marginals.npy')
        cap = np.load(f'{base_dir}day-ahead/system/capacity.npy')
        cov_prefix, freq, end = "miso_system", "1h", '2019-12-31 23:00'
    elif exp_level == "system_intra_1h":
        actuals = np.load(f'{base_dir}intra_hour/system/actuals.npy')
        marg = np.load(f'{base_dir}intra_hour/system/marginals_HA1.npy')
        cap = np.load(f'{base_dir}intra_hour/system/capacity.npy')
        cov_prefix, freq, end = "miso_system_IH", "15min", '2019-12-31 23:45'
    elif exp_level == "system_intra_2h":
        actuals = np.load(f'{base_dir}intra_hour/system/actuals.npy')
        marg = np.load(f'{base_dir}intra_hour/system/marginals_HA2.npy')
        cap = np.load(f'{base_dir}intra_hour/system/capacity.npy')
        cov_prefix, freq, end = "miso_system_IH", "15min", '2019-12-31 23:45'
    elif exp_level == "wind_system":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/MISO/wind/day-ahead/system/'
        actuals = np.load(f'{base_w}actuals.npy')
        marg = np.load(f'{base_w}marginals.npy')
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "miso_system", "1h", '2019-12-31 23:00'
    elif exp_level == "nyiso_system":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/NYISO/solar/day-ahead/system/'
        actuals = np.load(f'{base_w}actuals.npy')
        marg = np.load(f'{base_w}marginals.npy')
        cap = np.load(f'{base_w}capacity.npy')
        marg = marg.reshape(1, marg.shape[0], marg.shape[1])
        cov_prefix, freq, end = "nyiso_system", "1h", '2019-12-31 23:00'
    elif exp_level == "spp_system":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/day-ahead/system/'
        actuals = np.load(f'{base_w}actuals.npy')
        marg = np.load(f'{base_w}marginals.npy')
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "spp_system", "1h", '2019-12-31 23:00'
    elif exp_level == "spp_zones":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/day-ahead/zones/'
        actuals = np.load(f'{base_w}actuals.npy')
        marg = np.load(f'{base_w}marginals.npy')
        cap = np.load(f'{base_w}capacity.npy') 
        cov_prefix, freq, end = "spp_zones", "1h", '2019-12-31 23:00'
    elif exp_level == "spp_sites":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/day-ahead/sites/'
        actuals = np.load(f'{base_w}actuals.npy')
        marg = np.load(f'{base_w}marginals.npy')
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "spp_sites", "1h", '2019-12-31 23:00'
    elif exp_level == "ercot_system":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/ERCOT/solar/day-ahead/system/'
        actuals = np.load(f'{base_w}actuals.npy')
        marg = np.load(f'{base_w}marginals.npy')
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "ercot_system", "1h", '2018-12-31 23:00'
    elif exp_level == "ercot_sites":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/ERCOT/solar/day-ahead/sites/'
        actuals = np.load(f'{base_w}actuals.npy')
        marg = np.load(f'{base_w}marginals.npy')
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "ercot_sites", "1h", '2018-12-31 23:00'
    elif exp_level == "spp_copula":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/day-ahead/system/'
        actuals =  np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/day-ahead/system/actuals.npy')

        marg = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/spp_solar_copula.npy').reshape(1, 99, 8760)
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "spp_system", "1h", '2019-12-31 23:00'
    elif exp_level == "miso_copula":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/MISO/solar/day-ahead/system/'
        actuals =  np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/MISO/solar/day-ahead/system/actuals.npy')
        marg = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/miso_solar_copula.npy').reshape(1, 99, 8760)
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "miso_system", "1h", '2019-12-31 23:00'
    elif exp_level == "ercot_copula":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/ERCOT/solar/day-ahead/system/'
        actuals =  np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/ERCOT/solar/day-ahead/system/actuals.npy')
        marg = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/ercot_solar_copula.npy').reshape(1, 99, 8760)
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "ercot_system", "1h", '2018-12-31 23:00'

    elif exp_level == "spp_copula_zones":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/day-ahead/system/'
        actuals =  np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/day-ahead/system/actuals.npy')

        marg = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/spp_solar_all_zones_copula.npy').reshape(1, 99, 8760)
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "spp_system", "1h", '2019-12-31 23:00'

    elif exp_level == "miso_copula_zones":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/MISO/solar/day-ahead/system/'
        actuals =  np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/MISO/solar/day-ahead/system/actuals.npy')

        marg = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/miso_solar_all_zones_copula.npy').reshape(1, 99, 8760)
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "miso_system", "1h", '2019-12-31 23:00'

    elif exp_level == "ercot_copula_zones":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/ERCOT/solar/day-ahead/system/'
        actuals =  np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/ERCOT/solar/day-ahead/system/actuals.npy')
        marg = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/ercot_solar_all_zones_copula.npy').reshape(1, 99, 8760)
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "ercot_system", "1h", '2018-12-31 23:00'


    elif exp_level == "miso_zones_copula":
        actuals = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/zones_copula_actuals_miso.npy')
        marg = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/zones_copula_marginals_miso.npy')
        cap = np.load(f'{base_dir}day-ahead/zones/capacity.npy')
        cov_prefix, freq, end = "miso_zones", "1h", '2019-12-31 23:00'

    elif exp_level == "spp_zones_copula":
        base_w = '/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/day-ahead/zones/'
        actuals = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/zones_copula_actuals_miso.npy')
        marg = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/zones_copula_marginals_miso.npy')
        cap = np.load(f'{base_w}capacity.npy')
        cov_prefix, freq, end = "spp_zones", "1h", '2019-12-31 23:00'
    
    elif exp_level == "wind_test":
        actuals = np.load('/storage/home/hcoda1/1/amoradi30/scratch/wind/actual_test.npy').reshape(1,8760)
        marg = np.load('/storage/home/hcoda1/1/amoradi30/scratch/wind/pred_test.npy').reshape(1, 99, 8760)
        # cap = np.load('/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/nrel_forecasts/wind_test_capacity.npy')
        cap = np.max(actuals).reshape(1,1)
        cov_prefix, freq, end = "wind_test", "1h", '2019-12-31 23:00'

    elif exp_level == 'miso_load_system':
        actuals = np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/MISO/load/day-ahead/system/actuals.npy')
        marg = np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/MISO/load/day-ahead/system/marginals.npy')
        cap = np.max(actuals).reshape(1,1)
        cov_prefix, freq, end = "miso_load_system", "1h", '2019-12-31 23:00'
    elif exp_level == 'spp_load_system':
        actuals = np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/load/day-ahead/system/actuals.npy')
        marg = np.load('/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed/SPP/solar/load/day-ahead/system/marginals.npy')
        cap = np.max(actuals).reshape(1,1)
        cov_prefix, freq, end = "spp_load_system", "1h", '2019-12-31 23:00'
    else:
        raise ValueError(f"Invalid exp_level: {exp_level}")

    cov = None
    cov_blocks: Dict[str, slice] = {}

    if covariates:
        blocks = []
        start = 0
        for c in covariates:
            path = os.path.join(cov_base, f"{cov_prefix}_{c}.npy")
            if not os.path.exists(path):
                existing = sorted(
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(cov_base, f"{cov_prefix}_*.npy"))
                )
                raise FileNotFoundError(
                    f"Covariate file not found: {path}\n"
                    f"Available for prefix '{cov_prefix}':\n - " + "\n - ".join(existing)
                )
            arr = np.load(path)  # (N, k_c, T)
            k_c = arr.shape[1]
            blocks.append(arr)
            cov_blocks[c] = slice(start, start + k_c)
            
            start += k_c

        if blocks:
            cov_raw = np.concatenate(blocks, axis=1)  # (N, sum_k, T)
            cov = minmax_scale_3d(cov_raw)
            print('---')
            print(cov.shape)
            print('---')
    if exp_level == "ercot_system" or exp_level == "ercot_sites" or exp_level == "ercot_copula" or exp_level == "ercot_copula_zones":
        time_index = pd.date_range(start='2018-01-01', end=end, freq=freq)
    else:   
        time_index = pd.date_range(start='2019-01-01', end=end, freq=freq)

    if return_cov_map:
        return actuals, marg, cap, cov, time_index, cov_blocks
    else:
        return actuals, marg, cap, cov, time_index
