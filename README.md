# VN30 GARCH Volatility Forecasting

This repository contains a Time Series Analysis project on modeling and forecasting the daily volatility of the VN30 Index using GARCH-family models.

The main objective is to examine whether asymmetric GARCH models improve volatility modeling and forecasting compared with standard symmetric GARCH benchmarks.

---

## Project Summary

Daily VN30 closing prices from January 2015 to April 2026 are transformed into percentage log returns. The return series is then split chronologically into training, validation, and test samples to avoid look-ahead bias.

The project compares several volatility models, including ARCH, GARCH, GJR-GARCH, and EGARCH specifications with normal and Student-t innovations.

The key finding is:

> Asymmetric GARCH helps explain VN30 volatility dynamics in-sample, but it does not necessarily improve one-step-ahead volatility forecasting out-of-sample.

---

## Research Questions

1. Do VN30 returns exhibit stylized facts such as volatility clustering, fat tails, and ARCH effects?
2. Does GJR-GARCH better explain asymmetric volatility dynamics than symmetric GARCH?
3. Does the best in-sample model also produce the best validation forecasts?
4. Does the final selected model outperform simple volatility benchmarks?

---

## Repository Structure

```text
vn30-garch-volatility/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── tables/
├── reports/
├── src/
│   └── vn30_garch/
│       ├── data_eda.py
│       ├── garch_modeling.py
│       └── volatility_forecasting.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Data

The project uses daily VN30 closing prices from 2015-01-05 to 2026-04-29.

Daily log returns are computed as:

```text
r_t = 100 × [log(P_t) - log(P_{t-1})]
```

The processed return sample contains 2,823 observations.

The sample is split chronologically:

| Split      |                   Period | Observations |
| ---------- | -----------------------: | -----------: |
| Train      | 2015-01-06 to 2022-11-30 |        1,976 |
| Validation | 2022-12-01 to 2024-08-13 |          423 |
| Test       | 2024-08-14 to 2026-04-29 |          424 |

---

## Methodology

The analysis follows four main steps:

1. Clean VN30 price data and construct daily log returns.
2. Check time-series properties using ADF, Ljung-Box, and ARCH-LM tests.
3. Estimate and compare ARCH/GARCH-family models on the training sample.
4. Select the final model using validation QLIKE and evaluate it on the test sample.

Candidate models include:

* ARCH(1), ARCH(5)
* GARCH(1,1), GARCH(1,2), GARCH(2,1)
* AR(1)-GARCH(1,1)
* GJR-GARCH(1,1)
* AR(1)-GJR-GARCH(1,1)
* EGARCH(1,1)

Forecast performance is evaluated using QLIKE, MSE variance, MAE variance, and MAE volatility.

---

## Key Results

The best in-sample model is:

```text
AR(1)-GJR-GARCH(1,1)-t
```

This model captures volatility persistence, fat-tailed shocks, and asymmetric responses to negative returns.

However, the best validation forecasting model by QLIKE is:

```text
AR(1)-GARCH(1,1)-t
```

On the test sample, this model outperforms historical variance and rolling 21-day variance benchmarks by QLIKE.

| Model                   | Test QLIKE | MAE Volatility |
| ----------------------- | ---------: | -------------: |
| AR(1)-GARCH(1,1)-t      |     1.4123 |         0.7649 |
| Historical variance     |     1.5693 |         0.8095 |
| Rolling 21-day variance |     1.7618 |         0.7976 |

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/dchwy/vn30-garch-volatility.git
cd vn30-garch-volatility
```

Install dependencies:

```bash
uv sync
```

Run data cleaning and exploratory analysis:

```bash
uv run python src/vn30_garch/data_eda.py
```

Run GARCH model comparison:

```bash
uv run python src/vn30_garch/garch_modeling.py
```

Run validation and test forecasting:

```bash
uv run python src/vn30_garch/volatility_forecasting.py
```

---

## Outputs

Main outputs are saved in:

```text
outputs/figures/
outputs/tables/
reports/
```

Important files include:

* `outputs/tables/03_garch_model_comparison_train.csv`
* `outputs/tables/04_garch_diagnostics_train.csv`
* `outputs/tables/05_validation_forecast_performance.csv`
* `outputs/tables/06_test_forecast_performance.csv`
* `outputs/figures/06_test_volatility_forecast_comparison.png`

---

## Conclusion

The project shows that asymmetric volatility is important for interpreting VN30 risk dynamics. However, in the validation period, the simpler symmetric Student-t GARCH model performs slightly better for one-step-ahead forecasting.

This highlights the difference between a model that explains historical volatility well and a model that forecasts future volatility well.

---

## Author

Do Cong Huy
Class: DSEB 65B
Course: Time Series Analysis
National Economics University
