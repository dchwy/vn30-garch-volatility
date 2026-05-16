# Modeling and Forecasting the Volatility of the VN30 Index Using GARCH-family Models

This project models and forecasts the conditional volatility of daily VN30 index log-returns using GARCH-family models.

## Research Objective

The objective is to examine whether GARCH-family models can capture volatility clustering and improve volatility forecasting for the VN30 Index.

## Models

- GARCH(1,1)
- GJR-GARCH(1,1)
- EGARCH(1,1)

## Project Structure

```text
vn30-garch-volatility/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/vn30_garch/
├── outputs/
│   ├── figures/
│   └── tables/
├── reports/
├── scripts/
└── app/
```

## Environment Setup

```powershell
uv sync --extra dev
```

## Run Notebooks

```powershell
uv run jupyter notebook
```

## Data

Place the raw VN30 data file at:

```text
data/raw/VN30_2015_2026.csv
```

## Main Outputs

- `data/processed/vn30_returns.csv`
- `outputs/figures/`
- `outputs/tables/`
- `reports/VN30_GARCH_Report.pdf`
