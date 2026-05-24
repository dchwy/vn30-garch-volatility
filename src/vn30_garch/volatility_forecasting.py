"""
Out-of-sample VN30 volatility forecasting.

This script uses the clean train / valid / test split produced by data_eda.py.

Pipeline:
1. Load:
   - data/processed/vn30_returns_train.csv
   - data/processed/vn30_returns_valid.csv
   - data/processed/vn30_returns_test.csv

2. Fit candidate GARCH-family models on TRAIN.

3. Produce one-step-ahead volatility forecasts on VALID.

4. Select final GARCH model using validation QLIKE.

5. Refit selected model on TRAIN + VALID.

6. Produce one-step-ahead volatility forecasts on TEST.

7. Compare final GARCH model with benchmarks:
   - Historical variance
   - Rolling 21-day variance

Outputs:
   - outputs/tables/05_validation_forecast_performance.csv
   - outputs/tables/06_test_forecast_performance.csv
   - outputs/tables/07_test_volatility_forecasts.csv
   - outputs/figures/05_validation_qlike_comparison.png
   - outputs/figures/06_test_volatility_forecast_comparison.png

Recommended run:
    uv run python src/vn30_garch/volatility_forecasting.py
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from arch.univariate import arch_model
except ImportError as exc:
    raise ImportError(
        "Package 'arch' is required for GARCH forecasting. "
        "Install it with: uv add arch"
    ) from exc


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "vn30_returns_train.csv"
VALID_PATH = PROCESSED_DIR / "vn30_returns_valid.csv"
TEST_PATH = PROCESSED_DIR / "vn30_returns_test.csv"

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

VALID_PERFORMANCE_PATH = TABLE_DIR / "05_validation_forecast_performance.csv"
TEST_PERFORMANCE_PATH = TABLE_DIR / "06_test_forecast_performance.csv"
TEST_FORECAST_PATH = TABLE_DIR / "07_test_volatility_forecasts.csv"

VALID_QLIKE_FIGURE_PATH = FIGURE_DIR / "05_validation_qlike_comparison.png"
TEST_FORECAST_FIGURE_PATH = FIGURE_DIR / "06_test_volatility_forecast_comparison.png"


# ============================================================
# 2. MODEL SPECS
# ============================================================

@dataclass(frozen=True)
class ForecastModelSpec:
    """
    GARCH-family model specification for forecasting.
    """
    name: str
    mean: str
    lags: int
    vol: str
    p: int
    o: int
    q: int
    dist: str


CANDIDATE_MODELS = [
    ForecastModelSpec(
        name="GARCH(1,1)-t",
        mean="Constant",
        lags=0,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="t",
    ),
    ForecastModelSpec(
        name="AR(1)-GARCH(1,1)-t",
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="t",
    ),
    ForecastModelSpec(
        name="GJR-GARCH(1,1)-t",
        mean="Constant",
        lags=0,
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="t",
    ),
    ForecastModelSpec(
        name="AR(1)-GJR-GARCH(1,1)-t",
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="t",
    ),
]


# ============================================================
# 3. DATA LOADING
# ============================================================

def read_processed_split(path: Path, expected_split: str) -> pd.DataFrame:
    """
    Read one processed split file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split file not found: {path}\n"
            "Run data_eda.py first:\n"
            "    uv run python src/vn30_garch/data_eda.py"
        )

    df = pd.read_csv(path)

    required_columns = {"date", "return_pct"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing_columns)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")

    df = (
        df.dropna(subset=["date", "return_pct"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if "split" in df.columns:
        split_values = sorted(df["split"].dropna().unique().tolist())
        if split_values != [expected_split]:
            warnings.warn(
                f"Expected split '{expected_split}' in {path}, "
                f"but found {split_values}."
            )

    return df


def dataframe_to_return_series(df: pd.DataFrame) -> pd.Series:
    """
    Convert processed dataframe into return series indexed by date.
    """
    y = pd.Series(
        df["return_pct"].to_numpy(),
        index=pd.to_datetime(df["date"]),
        name="return_pct",
    )

    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.sort_index()

    return y


def load_all_splits(
    train_path: Path = TRAIN_PATH,
    valid_path: Path = VALID_PATH,
    test_path: Path = TEST_PATH,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load train, validation, and test return series.
    """
    train_df = read_processed_split(train_path, "train")
    valid_df = read_processed_split(valid_path, "valid")
    test_df = read_processed_split(test_path, "test")

    y_train = dataframe_to_return_series(train_df)
    y_valid = dataframe_to_return_series(valid_df)
    y_test = dataframe_to_return_series(test_df)

    print("[DATA]")
    print(f"Train: {len(y_train)} obs | {y_train.index.min().date()} -> {y_train.index.max().date()}")
    print(f"Valid: {len(y_valid)} obs | {y_valid.index.min().date()} -> {y_valid.index.max().date()}")
    print(f"Test:  {len(y_test)} obs | {y_test.index.min().date()} -> {y_test.index.max().date()}")
    print()

    return y_train, y_valid, y_test


# ============================================================
# 4. GARCH FITTING
# ============================================================

def fit_garch_model(y_fit: pd.Series, spec: ForecastModelSpec):
    """
    Fit one GARCH-family model on the provided fitting sample.
    """
    mean_lags = spec.lags if spec.mean.upper() == "AR" else 0

    model = arch_model(
        y_fit,
        mean=spec.mean,
        lags=mean_lags,
        vol=spec.vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        power=2.0,
        dist=spec.dist,
        rescale=False,
    )

    result = model.fit(
        disp="off",
        update_freq=0,
        show_warning=False,
    )

    return result


def get_last_valid_residual(result) -> float:
    """
    Extract the last valid residual from fitted model.
    """
    resid = pd.Series(result.resid)
    resid = resid.replace([np.inf, -np.inf], np.nan).dropna()

    if resid.empty:
        raise ValueError("No valid residuals found in fitted model.")

    return float(resid.iloc[-1])


def get_last_valid_variance(result) -> float:
    """
    Extract the last valid conditional variance from fitted model.
    """
    cond_vol = pd.Series(result.conditional_volatility)
    cond_vol = cond_vol.replace([np.inf, -np.inf], np.nan).dropna()

    if cond_vol.empty:
        raise ValueError("No valid conditional volatility found in fitted model.")

    return float(cond_vol.iloc[-1] ** 2)


def extract_ar1_parameter(params: pd.Series) -> float:
    """
    Extract AR(1) parameter robustly.

    In arch, AR(1) parameter can be named like:
        return_pct[1]
        y[1]
        None[1]
    depending on series name.
    """
    excluded_prefixes = ("alpha[", "beta[", "gamma[")
    excluded_names = {"omega", "mu", "Const", "nu", "eta", "lambda"}

    for name, value in params.items():
        name_str = str(name)

        if name_str in excluded_names:
            continue

        if name_str.startswith(excluded_prefixes):
            continue

        if re.search(r"\[1\]$", name_str):
            return float(value)

    return 0.0


def forecast_mean_one_step(
    params: pd.Series,
    spec: ForecastModelSpec,
    last_return: float,
) -> float:
    """
    Forecast conditional mean one step ahead.

    For Constant Mean:
        mean_t = mu

    For AR(1):
        mean_t = Const + phi * r_{t-1}
    """
    if spec.mean.upper() == "AR":
        const = float(params.get("Const", params.get("mu", 0.0)))
        phi = extract_ar1_parameter(params)
        return const + phi * last_return

    return float(params.get("mu", params.get("Const", 0.0)))


def forecast_variance_one_step(
    params: pd.Series,
    spec: ForecastModelSpec,
    last_residual: float,
    last_variance: float,
) -> float:
    """
    Forecast one-step-ahead conditional variance.

    For GARCH(1,1):
        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    For GJR-GARCH(1,1):
        sigma_t^2 = omega
                  + alpha * eps_{t-1}^2
                  + gamma * I(eps_{t-1} < 0) * eps_{t-1}^2
                  + beta * sigma_{t-1}^2
    """
    eps = 1e-12

    omega = float(params.get("omega", 0.0))
    alpha = float(params.get("alpha[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    gamma = float(params.get("gamma[1]", 0.0)) if spec.o > 0 else 0.0

    shock_sq = last_residual ** 2
    negative_indicator = 1.0 if last_residual < 0 else 0.0

    variance = (
        omega
        + alpha * shock_sq
        + gamma * negative_indicator * shock_sq
        + beta * last_variance
    )

    if not np.isfinite(variance) or variance <= eps:
        variance = eps

    return float(variance)


def recursive_one_step_variance_forecast(
    y_fit: pd.Series,
    y_eval: pd.Series,
    spec: ForecastModelSpec,
) -> tuple[pd.Series, object]:
    """
    Produce recursive one-step-ahead variance forecasts.

    The model parameters are estimated once on y_fit.
    Then forecasts are produced over y_eval.

    At each date t in y_eval:
    - Forecast variance using information up to t-1.
    - Observe actual return r_t.
    - Update residual and variance state for forecasting t+1.

    This avoids look-ahead bias because r_t is only used after its forecast.
    """
    result = fit_garch_model(y_fit, spec)

    params = result.params

    last_return = float(y_fit.iloc[-1])
    last_residual = get_last_valid_residual(result)
    last_variance = get_last_valid_variance(result)

    forecasts = pd.Series(index=y_eval.index, dtype=float, name=spec.name)

    for forecast_date, actual_return in y_eval.items():
        variance_forecast = forecast_variance_one_step(
            params=params,
            spec=spec,
            last_residual=last_residual,
            last_variance=last_variance,
        )

        forecasts.loc[forecast_date] = variance_forecast

        mean_forecast = forecast_mean_one_step(
            params=params,
            spec=spec,
            last_return=last_return,
        )

        current_residual = float(actual_return) - mean_forecast

        last_return = float(actual_return)
        last_residual = current_residual
        last_variance = variance_forecast

    return forecasts, result


# ============================================================
# 5. BENCHMARK FORECASTS
# ============================================================

def make_benchmark_variance_forecasts(
    y_all: pd.Series,
    eval_index: pd.Index,
    rolling_window: int = 21,
    min_periods_historical: int = 60,
) -> pd.DataFrame:
    """
    Create benchmark variance forecasts.

    Both benchmarks use shift(1), so forecast at date t only uses information up to t-1.
    """
    historical_var = (
        y_all.shift(1)
        .expanding(min_periods=min_periods_historical)
        .var()
    )

    rolling_var = (
        y_all.shift(1)
        .rolling(window=rolling_window, min_periods=rolling_window)
        .var()
    )

    benchmarks = pd.DataFrame(
        {
            "Historical variance": historical_var.loc[eval_index],
            "Rolling 21-day variance": rolling_var.loc[eval_index],
        },
        index=eval_index,
    )

    return benchmarks


# ============================================================
# 6. FORECAST EVALUATION
# ============================================================

def qlike_loss_series(
    realized_variance: pd.Series,
    forecast_variance: pd.Series,
) -> pd.Series:
    """
    Compute QLIKE loss series.

    QLIKE = log(forecast variance) + realized variance / forecast variance

    Lower is better.
    """
    eps = 1e-12

    realized = pd.Series(realized_variance).astype(float)
    forecast = pd.Series(forecast_variance).astype(float)

    df = pd.DataFrame(
        {
            "realized": realized,
            "forecast": forecast,
        }
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["forecast"] > eps]

    loss = pd.Series(index=forecast.index, dtype=float)
    loss.loc[df.index] = np.log(df["forecast"]) + df["realized"] / df["forecast"]

    return loss


def evaluate_forecast(
    realized_variance: pd.Series,
    forecast_variance: pd.Series,
    model_name: str,
    sample: str,
    is_garch_candidate: bool,
) -> dict:
    """
    Evaluate one variance forecast.
    """
    eps = 1e-12

    df = pd.DataFrame(
        {
            "realized_variance": realized_variance,
            "forecast_variance": forecast_variance,
        }
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["forecast_variance"] > eps]

    if df.empty:
        return {
            "sample": sample,
            "model": model_name,
            "is_garch_candidate": is_garch_candidate,
            "n_eval": 0,
            "mse_variance": np.nan,
            "mae_variance": np.nan,
            "rmse_variance": np.nan,
            "qlike": np.nan,
            "mse_volatility": np.nan,
            "mae_volatility": np.nan,
        }

    realized_var = df["realized_variance"]
    forecast_var = df["forecast_variance"]

    realized_vol = np.sqrt(realized_var)
    forecast_vol = np.sqrt(forecast_var)

    mse_var = float(np.mean((realized_var - forecast_var) ** 2))
    mae_var = float(np.mean(np.abs(realized_var - forecast_var)))

    mse_vol = float(np.mean((realized_vol - forecast_vol) ** 2))
    mae_vol = float(np.mean(np.abs(realized_vol - forecast_vol)))

    qlike = float(
        qlike_loss_series(realized_var, forecast_var)
        .dropna()
        .mean()
    )

    return {
        "sample": sample,
        "model": model_name,
        "is_garch_candidate": is_garch_candidate,
        "n_eval": int(len(df)),
        "mse_variance": mse_var,
        "mae_variance": mae_var,
        "rmse_variance": float(np.sqrt(mse_var)),
        "qlike": qlike,
        "mse_volatility": mse_vol,
        "mae_volatility": mae_vol,
    }


def evaluate_forecast_table(
    realized_variance: pd.Series,
    forecast_dict: dict[str, pd.Series],
    sample: str,
    garch_candidate_names: set[str],
) -> pd.DataFrame:
    """
    Evaluate multiple forecasts.
    """
    rows = []

    for model_name, forecast in forecast_dict.items():
        rows.append(
            evaluate_forecast(
                realized_variance=realized_variance,
                forecast_variance=forecast,
                model_name=model_name,
                sample=sample,
                is_garch_candidate=model_name in garch_candidate_names,
            )
        )

    performance = pd.DataFrame(rows)

    metric_cols = [
        "mse_variance",
        "mae_variance",
        "rmse_variance",
        "qlike",
        "mse_volatility",
        "mae_volatility",
    ]

    for col in metric_cols:
        performance[f"{col}_rank"] = performance[col].rank(method="min")

    performance = performance.sort_values(["qlike", "mae_variance"])

    return performance


def select_final_garch_model(validation_performance: pd.DataFrame) -> str:
    """
    Select final GARCH model using validation QLIKE.

    Benchmarks are not selected as final model; they are comparison baselines.
    """
    candidates = validation_performance[
        validation_performance["is_garch_candidate"] == True  # noqa: E712
    ].copy()

    candidates = candidates.dropna(subset=["qlike", "mae_variance"])

    if candidates.empty:
        raise RuntimeError("No valid GARCH candidate available for selection.")

    selected = candidates.sort_values(["qlike", "mae_variance"]).iloc[0]

    selected_model = str(selected["model"])

    print()
    print("[VALIDATION MODEL SELECTION]")
    print(f"Selected final GARCH model: {selected_model}")
    print(f"Validation QLIKE: {selected['qlike']:.6f}")
    print(f"Validation MAE variance: {selected['mae_variance']:.6f}")
    print()

    return selected_model


# ============================================================
# 7. FORECAST DATAFRAMES
# ============================================================

def make_forecast_dataframe(
    y_eval: pd.Series,
    selected_model_name: str,
    selected_model_variance_forecast: pd.Series,
    benchmark_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build compact forecast dataframe for reporting and plotting.
    """
    df = pd.DataFrame(index=y_eval.index)

    df["return_pct"] = y_eval
    df["realized_variance_proxy"] = df["return_pct"] ** 2
    df["realized_volatility_proxy"] = df["return_pct"].abs()

    df[f"{selected_model_name}_variance_forecast"] = selected_model_variance_forecast
    df["Historical variance_forecast"] = benchmark_forecasts["Historical variance"]
    df["Rolling 21-day variance_forecast"] = benchmark_forecasts["Rolling 21-day variance"]

    df[f"{selected_model_name}_volatility_forecast"] = np.sqrt(
        df[f"{selected_model_name}_variance_forecast"]
    )

    df["Historical volatility_forecast"] = np.sqrt(
        df["Historical variance_forecast"]
    )

    df["Rolling 21-day volatility_forecast"] = np.sqrt(
        df["Rolling 21-day variance_forecast"]
    )

    qlike_selected = qlike_loss_series(
        df["realized_variance_proxy"],
        df[f"{selected_model_name}_variance_forecast"],
    )

    qlike_historical = qlike_loss_series(
        df["realized_variance_proxy"],
        df["Historical variance_forecast"],
    )

    qlike_rolling = qlike_loss_series(
        df["realized_variance_proxy"],
        df["Rolling 21-day variance_forecast"],
    )

    df[f"{selected_model_name}_qlike_loss"] = qlike_selected
    df["Historical variance_qlike_loss"] = qlike_historical
    df["Rolling 21-day variance_qlike_loss"] = qlike_rolling

    return df


# ============================================================
# 8. FIGURES
# ============================================================

def plot_validation_qlike(
    validation_performance: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot validation QLIKE across GARCH candidates and benchmarks.
    """
    plot_df = validation_performance.dropna(subset=["qlike"]).copy()
    plot_df = plot_df.sort_values("qlike")

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df["model"], plot_df["qlike"])
    plt.title("Validation Volatility Forecast Performance by QLIKE")
    plt.xlabel("Model")
    plt.ylabel("QLIKE loss, lower is better")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_test_forecast_comparison(
    test_forecast_df: pd.DataFrame,
    selected_model_name: str,
    output_path: Path,
) -> None:
    """
    Plot realized volatility proxy and test volatility forecasts.
    """
    selected_col = f"{selected_model_name}_volatility_forecast"

    plot_df = test_forecast_df.dropna(
        subset=[
            "realized_volatility_proxy",
            selected_col,
            "Historical volatility_forecast",
            "Rolling 21-day volatility_forecast",
        ]
    ).copy()

    plt.figure(figsize=(13, 6))

    plt.plot(
        plot_df.index,
        plot_df["realized_volatility_proxy"],
        linewidth=0.7,
        label="Realized volatility proxy |r_t|",
    )

    plt.plot(
        plot_df.index,
        plot_df[selected_col],
        linewidth=1.3,
        label=selected_model_name,
    )

    plt.plot(
        plot_df.index,
        plot_df["Rolling 21-day volatility_forecast"],
        linewidth=1.0,
        label="Rolling 21-day benchmark",
    )

    plt.plot(
        plot_df.index,
        plot_df["Historical volatility_forecast"],
        linewidth=1.0,
        label="Historical variance benchmark",
    )

    plt.title("Test Sample One-step-ahead Volatility Forecasts")
    plt.xlabel("Date")
    plt.ylabel("Volatility, percent")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(
    validation_performance: pd.DataFrame,
    test_performance: pd.DataFrame,
    test_forecast_df: pd.DataFrame,
    selected_model_name: str,
) -> None:
    """
    Save tables and figures.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    validation_out = validation_performance.copy()
    validation_out["selected_final_garch_model"] = (
        validation_out["model"] == selected_model_name
    )

    test_out = test_performance.copy()
    test_out["selected_final_garch_model"] = (
        test_out["model"] == selected_model_name
    )

    validation_out.to_csv(VALID_PERFORMANCE_PATH, index=False)
    test_out.to_csv(TEST_PERFORMANCE_PATH, index=False)

    forecast_out = test_forecast_df.reset_index()
    forecast_out = forecast_out.rename(columns={forecast_out.columns[0]: "date"})
    forecast_out.to_csv(TEST_FORECAST_PATH, index=False)

    plot_validation_qlike(
        validation_performance=validation_out,
        output_path=VALID_QLIKE_FIGURE_PATH,
    )

    plot_test_forecast_comparison(
        test_forecast_df=test_forecast_df,
        selected_model_name=selected_model_name,
        output_path=TEST_FORECAST_FIGURE_PATH,
    )

    print("[SAVED OUTPUTS]")
    print(f"Table:  {VALID_PERFORMANCE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Table:  {TEST_PERFORMANCE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Table:  {TEST_FORECAST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {VALID_QLIKE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {TEST_FORECAST_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print()


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def run_forecasting_pipeline(
    train_path: Path = TRAIN_PATH,
    valid_path: Path = VALID_PATH,
    test_path: Path = TEST_PATH,
) -> dict:
    """
    Run complete validation and test volatility forecasting pipeline.
    """
    y_train, y_valid, y_test = load_all_splits(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
    )

    y_train_valid = pd.concat([y_train, y_valid]).sort_index()
    y_all = pd.concat([y_train, y_valid, y_test]).sort_index()

    garch_candidate_names = {spec.name for spec in CANDIDATE_MODELS}

    print("[STEP 1] Validation forecasts from models fitted on TRAIN")
    validation_forecasts: dict[str, pd.Series] = {}
    validation_fit_results: dict[str, object] = {}

    for spec in CANDIDATE_MODELS:
        print(f"[VALID FORECAST] {spec.name}")

        try:
            forecast, result = recursive_one_step_variance_forecast(
                y_fit=y_train,
                y_eval=y_valid,
                spec=spec,
            )

            validation_forecasts[spec.name] = forecast
            validation_fit_results[spec.name] = result

        except Exception as exc:
            warnings.warn(f"Validation forecast failed for {spec.name}. Reason: {exc}")

    valid_benchmarks = make_benchmark_variance_forecasts(
        y_all=pd.concat([y_train, y_valid]).sort_index(),
        eval_index=y_valid.index,
    )

    validation_forecasts["Historical variance"] = valid_benchmarks["Historical variance"]
    validation_forecasts["Rolling 21-day variance"] = valid_benchmarks[
        "Rolling 21-day variance"
    ]

    validation_performance = evaluate_forecast_table(
        realized_variance=y_valid ** 2,
        forecast_dict=validation_forecasts,
        sample="valid",
        garch_candidate_names=garch_candidate_names,
    )

    selected_model_name = select_final_garch_model(validation_performance)

    print("[VALIDATION PERFORMANCE]")
    print(
        validation_performance[
            [
                "model",
                "n_eval",
                "mse_variance",
                "mae_variance",
                "qlike",
                "mae_volatility",
            ]
        ].to_string(index=False)
    )
    print()

    selected_spec = next(
        spec for spec in CANDIDATE_MODELS
        if spec.name == selected_model_name
    )

    print("[STEP 2] Final TEST forecast from selected model fitted on TRAIN + VALID")
    print(f"[FINAL MODEL] {selected_model_name}")

    test_selected_forecast, final_fit_result = recursive_one_step_variance_forecast(
        y_fit=y_train_valid,
        y_eval=y_test,
        spec=selected_spec,
    )

    test_benchmarks = make_benchmark_variance_forecasts(
        y_all=y_all,
        eval_index=y_test.index,
    )

    test_forecasts = {
        selected_model_name: test_selected_forecast,
        "Historical variance": test_benchmarks["Historical variance"],
        "Rolling 21-day variance": test_benchmarks["Rolling 21-day variance"],
    }

    test_performance = evaluate_forecast_table(
        realized_variance=y_test ** 2,
        forecast_dict=test_forecasts,
        sample="test",
        garch_candidate_names={selected_model_name},
    )

    test_forecast_df = make_forecast_dataframe(
        y_eval=y_test,
        selected_model_name=selected_model_name,
        selected_model_variance_forecast=test_selected_forecast,
        benchmark_forecasts=test_benchmarks,
    )

    print("[TEST PERFORMANCE]")
    print(
        test_performance[
            [
                "model",
                "n_eval",
                "mse_variance",
                "mae_variance",
                "qlike",
                "mae_volatility",
            ]
        ].to_string(index=False)
    )
    print()

    save_outputs(
        validation_performance=validation_performance,
        test_performance=test_performance,
        test_forecast_df=test_forecast_df,
        selected_model_name=selected_model_name,
    )

    return {
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "validation_forecasts": validation_forecasts,
        "validation_performance": validation_performance,
        "selected_model_name": selected_model_name,
        "final_fit_result": final_fit_result,
        "test_forecast_df": test_forecast_df,
        "test_performance": test_performance,
    }


# ============================================================
# 11. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Validation and test volatility forecasting for VN30.",
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default=str(TRAIN_PATH),
        help="Path to processed train csv.",
    )

    parser.add_argument(
        "--valid-path",
        type=str,
        default=str(VALID_PATH),
        help="Path to processed validation csv.",
    )

    parser.add_argument(
        "--test-path",
        type=str,
        default=str(TEST_PATH),
        help="Path to processed test csv.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run from terminal:

        uv run python src/vn30_garch/volatility_forecasting.py
    """
    args = parse_args()

    run_forecasting_pipeline(
        train_path=Path(args.train_path),
        valid_path=Path(args.valid_path),
        test_path=Path(args.test_path),
    )


if __name__ == "__main__":
    main()