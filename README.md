# Context-Aware Conformal Prediction

Conformal prediction framework for probabilistic energy forecasting, evaluated on the [NREL PERFORM](https://github.com/nrel/PERFORM) benchmark dataset. Extends standard Conformal Quantile Regression (CQR) with context-aware methods that adapt prediction intervals based on covariates, temporal patterns, and learned representations.

---

## Methods

| Key | Class | Description |
|-----|-------|-------------|
| `nrel` | — | Pass-through baseline: raw NREL PERFORM marginal quantiles (no CP adjustment) |
| `cqr` | `CQRConformalizer` | Standard Conformal Quantile Regression — uniform quantile adjustment |
| `nexcp` | `NExCPConformalizer` | NExCP — exponentially decaying weights favor recent calibration errors |
| `adaptive` | `AdaptiveConformalizer` | ACI-style online adaptation — adjusts coverage level $\alpha_t$ using feedback |
| `kmeans` | `KMeansConformalizer` | Clusters calibration covariates; applies cluster-specific quantiles |
| `knn` | `KNNConformalizer` | $k$-nearest-neighbor weighted conformity quantiles |
| `kernel` | `KernelConformalizer` | Kernel-weighted conformity quantiles (RBF, Laplacian, linear, poly) |
| `hopcpt` | `HopCPTLearnedConformalizer` | Learns covariate embeddings via Hopfield/attention; weighted conformity quantiles |

All methods share a unified `BaseConformalizer` interface: `fit(actual, forecast, past_cov)` + `batch_forecast(forecasts, future_cov)`.

---

## Project Structure

```
conformal/          # Conformalizer implementations
  base.py           # BaseConformalizer, FitData, quantile utilities
  cqr.py
  nexcp.py
  adaptive.py
  kmeans.py
  knn.py
  kernel.py
  hopcpt_learned.py
  factory.py        # make_conformalizer() + SEARCH_SPACES

dataio/
  loaders.py        # load_miso_like() — loads NREL PERFORM arrays + covariates

eval/
  metrics.py        # winkler_score(), picp_aiw()
  splitter.py       # rolling_windows(), split_train_val()
  tuning.py         # search_winkler() — grid/random HP search with CSV logging

runner/
  experiment.py     # run_one_site(), run_all_sites(), aggregate_sites()

main.py             # CLI: run experiment, print aggregated metrics table
run_experiment_csv.py  # CLI: export per-timestep prediction intervals to CSV
```

---

## Data

Experiments use the **NREL PERFORM** dataset — hourly day-ahead probabilistic forecasts (99 quantiles) for solar, wind, and load across four US balancing areas (MISO, SPP, ERCOT, NYISO), at system, zone, and site aggregation levels.

Data is not included in this repository. Set the paths in `dataio/loaders.py` to point to your local copy of the processed NREL PERFORM arrays.

**Expected directory layout:**
```
<NREL_PERFORM_ROOT>/processed/
  MISO/solar/day-ahead/system/{actuals,marginals,capacity}.npy
  MISO/solar/day-ahead/zones/{actuals,marginals,capacity}.npy
  MISO/solar/day-ahead/sites/{actuals,marginals,capacity}.npy
  MISO/load/day-ahead/system/{actuals,marginals}.npy
  MISO/wind/day-ahead/system/{actuals,marginals,capacity}.npy
  MISO/solar/intra_hour/system/{actuals,marginals_HA1,marginals_HA2,capacity}.npy
  SPP/solar/day-ahead/{system,zones,sites}/...
  ERCOT/solar/day-ahead/{system,sites}/...
  NYISO/solar/day-ahead/system/...
  cov/
    {prefix}_{time,weather,historical,solarity}.npy   # (N_sites, D, T)
```

**Array shapes** (day-ahead hourly, T = 8 760 per year):
- `actuals.npy` — `(N, T)` actual values
- `marginals.npy` — `(N, 99, T)` forecast quantiles at the 1st–99th percentiles
- `capacity.npy` — `(N, 1)` installed capacity (used for normalization)
- covariate files — `(N, D, T)` with D features per covariate block

---

## Installation

```bash
git clone https://github.com/<your-org>/Context-aware-Conformal-Prediction.git
cd Context-aware-Conformal-Prediction
pip install -r requirements.txt
```

`torch` is only required for the `hopcpt` and `fea` methods. All other methods depend only on `numpy`, `scikit-learn`, and `scipy`.

---

## Usage

### Run a full experiment (print metrics table)

```bash
python main.py \
  --exp_level miso_system \
  --methods NREL CQR NexCP KMeans KNN Kernel HopCPT \
  --covariates time weather \
  --test_window_days 7 \
  --flavor solar \
  --out_file results_miso_system.csv
```

### Export per-timestep prediction intervals to CSV

```bash
python run_experiment_csv.py \
  --exp_level miso_system \
  --methods NREL CQR KNN \
  --level 0.9 \
  --covariates time weather \
  --out_csv intervals_miso.csv
```

### Available `--exp_level` values

| Value | Dataset | Resource | Aggregation |
|-------|---------|----------|-------------|
| `miso_system` | MISO 2019 | Solar | System total |
| `miso_zones` | MISO 2019 | Solar | Zones |
| `miso_sites` | MISO 2019 | Solar | Individual sites |
| `miso_load_system` | MISO 2019 | Load | System total |
| `wind_system` | MISO 2019 | Wind | System total |
| `system_intra_1h` | MISO 2019 | Solar | System (HA+1h) |
| `system_intra_2h` | MISO 2019 | Solar | System (HA+2h) |
| `spp_system` | SPP 2019 | Solar | System total |
| `spp_zones` | SPP 2019 | Solar | Zones |
| `spp_sites` | SPP 2019 | Solar | Individual sites |
| `ercot_system` | ERCOT 2018 | Solar | System total |
| `ercot_sites` | ERCOT 2018 | Solar | Individual sites |
| `nyiso_system` | NYISO 2019 | Solar | System total |

### Available `--covariates` blocks

| Block | Features | Shape |
|-------|----------|-------|
| `time` | Hour-of-day, day-of-year, day-of-week (sin/cos encoded) | D=6 |
| `weather` | ECMWF forecast variables (irradiance, temperature, wind, …) | D=60 |
| `historical` | Lagged actual values | D=24 |
| `solarity` | Solar geometry (elevation, azimuth) | D=2 |

Pass multiple blocks to concatenate them: `--covariates time weather`.

---

## Programmatic API

```python
import numpy as np
from conformal.factory import make_conformalizer
from eval.metrics import picp_aiw, winkler_score

# actual: (T,)  forecast: (99, T)  cov: (T, D)
conf = make_conformalizer("knn", nneighbors=200, value_bounds=(0.0, 1.0))
conf.fit(y_cal, Q_cal, past_cov=cov_cal)

pred = conf.batch_forecast(Q_test, future_cov=cov_test)  # (T_test, 99)

picp, aiw = picp_aiw(y_test, pred)
ws = winkler_score(y_test, pred)
```

### Hyperparameter search

```python
from eval.tuning import search_winkler
from conformal.factory import SEARCH_SPACES

best_params, best_score = search_winkler(
    method_name="knn",
    param_grid=SEARCH_SPACES["knn"],          # {"nneighbors": [50, 100, 200, 500, 1000]}
    y_train=y_cal, Q_train=Q_cal, cov_train=cov_cal,
    y_val=y_val,   Q_val=Q_val,   cov_val=cov_val,
    strategy="grid",
    value_bounds=(0.0, 1.0),
)
conf = make_conformalizer("knn", **best_params, value_bounds=(0.0, 1.0))
```

---

## Evaluation Protocol

- **Rolling windows**: calibration starts at the year boundary; the test window advances by `--test_window_days` each step.
- **Validation split**: the last 7 days of the calibration period are held out for hyperparameter tuning.
- **Normalization**: all values are divided by site capacity before fitting and restored afterward.
- **Solar masking**: zero-generation hours (before sunrise / after sunset) are excluded from calibration and test (`--flavor solar`). Wind and load use all hours (`--flavor wind` / `--flavor load`).
- **Metrics** (reported per coverage level 90 / 80 / 70 / 60 %):
  - **PICP** — prediction interval coverage probability
  - **AIW** — average interval width
  - **Winkler score** — width + miscoverage penalty, lower is better


