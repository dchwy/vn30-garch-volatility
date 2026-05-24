"""
GARCH-family model selection and diagnostics for VN30 volatility.

This script uses TRAINING DATA ONLY.

Pipeline:
1. Load data/processed/vn30_returns_train.csv.
2. Fit candidate ARCH/GARCH-family models on the training sample.
3. Compare models using log-likelihood, AIC, BIC, and key parameters.
4. Run residual diagnostics on standardized residuals.
5. Select the best model on the training sample only.
6. Save compact report outputs:
   - outputs/tables/03_garch_model_comparison_train.csv
   - outputs/tables/04_garch_diagnostics_train.csv
   - outputs/figures/03_conditional_volatility_train.png
   - outputs/figures/04_standardized_residual_diagnostics_train.png

Recommended run:
    uv run python src/vn30_garch/garch_modeling.py

"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

try:
    from arch.univariate import arch_model
except ImportError as exc:
    raise ImportError(
        "Package 'arch' is required for GARCH modeling. "
        "Install it with: uv add arch"
    ) from exc


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "vn30_returns_train.csv"

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

COMPARISON_PATH = TABLE_DIR / "03_garch_model_comparison_train.csv"
DIAGNOSTICS_PATH = TABLE_DIR / "04_garch_diagnostics_train.csv"

VOL_FIGURE_PATH = FIGURE_DIR / "03_conditional_volatility_train.png"
RESID_FIGURE_PATH = FIGURE_DIR / "04_standardized_residual_diagnostics_train.png"


# ============================================================
# 2. MODEL SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class ModelSpec:
    """
    Candidate GARCH-family model specification.

    Parameters
    ----------
    name:
        Display name of the model.

    mean:
        Mean equation type used by arch_model.
        Common values: "Constant", "AR".

    lags:
        Number of AR lags in the mean equation.
        Used only when mean="AR".

    vol:
        Volatility model type.
        Common values: "ARCH", "GARCH", "EGARCH".

    p, o, q:
        Volatility equation orders.
        For GJR-GARCH(1,1), use vol="GARCH", p=1, o=1, q=1.
    """
    name: str
    mean: str
    lags: int
    vol: str
    p: int
    o: int
    q: int


CONSTANT_MEAN_SPECS = [
    ModelSpec("ARCH(1)", "Constant", 0, "ARCH", 1, 0, 0),
    ModelSpec("ARCH(5)", "Constant", 0, "ARCH", 5, 0, 0),
    ModelSpec("GARCH(1,1)", "Constant", 0, "GARCH", 1, 0, 1),
    ModelSpec("GARCH(1,2)", "Constant", 0, "GARCH", 1, 0, 2),
    ModelSpec("GARCH(2,1)", "Constant", 0, "GARCH", 2, 0, 1),
    ModelSpec("EGARCH(1,1)", "Constant", 0, "EGARCH", 1, 0, 1),
    ModelSpec("GJR-GARCH(1,1)", "Constant", 0, "GARCH", 1, 1, 1),
]

AR_MEAN_SPECS = [
    ModelSpec("AR(1)-GARCH(1,1)", "AR", 1, "GARCH", 1, 0, 1),
    ModelSpec("AR(1)-GJR-GARCH(1,1)", "AR", 1, "GARCH", 1, 1, 1),
]

DISTRIBUTIONS = ["normal", "t"]


# ============================================================
# 3. DATA LOADING
# ============================================================

def read_training_data(data_path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """
    Read processed training dataset.

    Expected columns from data_eda.py:
        date, close, return_pct, squared_return, abs_return, split, ...
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data file not found: {data_path}\n"
            "Run data_eda.py first:\n"
            "    uv run python src/vn30_garch/data_eda.py"
        )

    df = pd.read_csv(data_path)

    required_columns = {"date", "return_pct"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in training data: {sorted(missing_columns)}\n"
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
        unique_splits = sorted(df["split"].dropna().unique().tolist())
        if unique_splits != ["train"]:
            warnings.warn(
                "Input data contains split labels other than 'train': "
                f"{unique_splits}. This script should use training data only."
            )

    if len(df) < 300:
        warnings.warn(
            f"Training sample has only {len(df)} observations. "
            "GARCH estimates may be unstable."
        )

    return df


def load_training_return_series(data_path: Path = TRAIN_DATA_PATH) -> pd.Series:
    """
    Load VN30 training return series in percentage points.
    """
    df = read_training_data(data_path)

    y = pd.Series(
        df["return_pct"].to_numpy(),
        index=df["date"],
        name="return_pct",
    )

    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    if y.empty:
        raise ValueError("Training return series is empty after cleaning.")

    print("[DATA]")
    print(f"File: {data_path}")
    print(f"Sample: TRAIN ONLY")
    print(f"Observations: {len(y)}")
    print(f"Date range: {y.index.min().date()} -> {y.index.max().date()}")
    print(f"Mean return: {y.mean():.6f}%")
    print(f"Std. dev.:   {y.std():.6f}%")
    print()

    return y


# ============================================================
# 4. MODEL FITTING HELPERS
# ============================================================

def fit_one_model(y: pd.Series, spec: ModelSpec, dist: str):
    """
    Fit one ARCH/GARCH-family model.
    """
    mean_lags = spec.lags if spec.mean.upper() == "AR" else 0

    model = arch_model(
        y,
        mean=spec.mean,
        lags=mean_lags,
        vol=spec.vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        power=2.0,
        dist=dist,
        rescale=False,
    )

    result = model.fit(
        disp="off",
        update_freq=0,
        show_warning=False,
    )

    return result


def param_value(result, name: str) -> float:
    """
    Safely extract parameter value.
    """
    if name in result.params.index:
        return float(result.params[name])
    return np.nan


def param_pvalue(result, name: str) -> float:
    """
    Safely extract parameter p-value.
    """
    if name in result.pvalues.index:
        return float(result.pvalues[name])
    return np.nan


def sum_params_by_prefix(result, prefix: str) -> float:
    """
    Sum parameters whose names start with a prefix.

    Example:
        alpha[1], alpha[2], ...
    """
    values = [
        float(value)
        for name, value in result.params.items()
        if str(name).startswith(prefix)
    ]

    return float(np.sum(values)) if values else np.nan


def make_comparison_row(spec: ModelSpec, dist: str, result) -> dict:
    """
    Create one model comparison row.
    """
    alpha_sum = sum_params_by_prefix(result, "alpha[")
    beta_sum = sum_params_by_prefix(result, "beta[")
    gamma_sum = sum_params_by_prefix(result, "gamma[")

    garch_persistence = (
        alpha_sum + beta_sum
        if not np.isnan(alpha_sum) and not np.isnan(beta_sum)
        else np.nan
    )

    # Common approximation for GJR-GARCH persistence under symmetric shocks:
    # alpha + beta + gamma / 2
    gjr_persistence_approx = (
        alpha_sum + beta_sum + 0.5 * gamma_sum
        if not np.isnan(alpha_sum)
        and not np.isnan(beta_sum)
        and not np.isnan(gamma_sum)
        else np.nan
    )

    return {
        "sample": "train",
        "model": f"{spec.name}-{dist}",
        "base_model": spec.name,
        "mean_model": spec.mean,
        "mean_lags": spec.lags,
        "vol_model": spec.vol,
        "p": spec.p,
        "o": spec.o,
        "q": spec.q,
        "distribution": dist,
        "nobs": int(result.nobs),
        "loglikelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "convergence_flag": int(getattr(result, "convergence_flag", -1)),
        "omega": param_value(result, "omega"),
        "alpha_sum": alpha_sum,
        "beta_sum": beta_sum,
        "gamma_sum": gamma_sum,
        "garch_persistence_alpha_plus_beta": garch_persistence,
        "gjr_persistence_approx": gjr_persistence_approx,
        "nu_student_t": param_value(result, "nu"),
        "mu": param_value(result, "mu"),
        "const": param_value(result, "Const"),
        "ar1": param_value(result, "return_pct[1]"),
        "omega_pvalue": param_pvalue(result, "omega"),
        "alpha1_pvalue": param_pvalue(result, "alpha[1]"),
        "beta1_pvalue": param_pvalue(result, "beta[1]"),
        "gamma1_pvalue": param_pvalue(result, "gamma[1]"),
        "nu_pvalue": param_pvalue(result, "nu"),
        "ar1_pvalue": param_pvalue(result, "return_pct[1]"),
    }


# ============================================================
# 5. RESIDUAL DIAGNOSTICS
# ============================================================

def get_standardized_residuals(result) -> pd.Series:
    """
    Extract standardized residuals and drop invalid values.
    """
    std_resid = pd.Series(result.std_resid)
    std_resid = std_resid.replace([np.inf, -np.inf], np.nan).dropna()

    return std_resid


def run_diagnostics(result, model_name: str, max_lag: int = 20) -> dict:
    """
    Run residual diagnostics.

    A good GARCH model should leave:
    - standardized residuals with little/no autocorrelation
    - squared standardized residuals with little/no autocorrelation
    - no remaining ARCH effects
    """
    std_resid = get_standardized_residuals(result)

    if len(std_resid) < 30:
        raise ValueError(f"Too few residuals for diagnostics: {model_name}")

    lag = min(max_lag, max(1, len(std_resid) // 5))

    lb_resid = acorr_ljungbox(
        std_resid,
        lags=[lag],
        return_df=True,
    )

    lb_sq = acorr_ljungbox(
        std_resid**2,
        lags=[lag],
        return_df=True,
    )

    arch_lm_stat, arch_lm_pvalue, arch_f_stat, arch_f_pvalue = het_arch(
        std_resid,
        nlags=lag,
    )

    return {
        "sample": "train",
        "model": model_name,
        "diagnostic_lag": lag,
        "lb_std_resid_stat": float(lb_resid["lb_stat"].iloc[0]),
        "lb_std_resid_pvalue": float(lb_resid["lb_pvalue"].iloc[0]),
        "lb_squared_std_resid_stat": float(lb_sq["lb_stat"].iloc[0]),
        "lb_squared_std_resid_pvalue": float(lb_sq["lb_pvalue"].iloc[0]),
        "arch_lm_stat": float(arch_lm_stat),
        "arch_lm_pvalue": float(arch_lm_pvalue),
        "arch_f_stat": float(arch_f_stat),
        "arch_f_pvalue": float(arch_f_pvalue),
        "pass_lb_std_resid_5pct": float(lb_resid["lb_pvalue"].iloc[0]) > 0.05,
        "pass_lb_squared_std_resid_5pct": float(lb_sq["lb_pvalue"].iloc[0]) > 0.05,
        "pass_arch_lm_5pct": float(arch_lm_pvalue) > 0.05,
    }


# ============================================================
# 6. FIT ALL CANDIDATE MODELS
# ============================================================

def fit_all_models(
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Fit all candidate models on the training sample.
    """
    specs = CONSTANT_MEAN_SPECS + AR_MEAN_SPECS

    comparison_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    fitted_results: dict[str, object] = {}

    for spec in specs:
        for dist in DISTRIBUTIONS:
            model_name = f"{spec.name}-{dist}"
            print(f"[FIT] {model_name}")

            try:
                result = fit_one_model(y, spec, dist)

                comparison_rows.append(
                    make_comparison_row(spec, dist, result)
                )

                diagnostic_rows.append(
                    run_diagnostics(result, model_name)
                )

                fitted_results[model_name] = result

            except Exception as exc:
                warnings.warn(f"Model failed: {model_name}. Reason: {exc}")

                comparison_rows.append(
                    {
                        "sample": "train",
                        "model": model_name,
                        "base_model": spec.name,
                        "mean_model": spec.mean,
                        "mean_lags": spec.lags,
                        "vol_model": spec.vol,
                        "p": spec.p,
                        "o": spec.o,
                        "q": spec.q,
                        "distribution": dist,
                        "nobs": np.nan,
                        "loglikelihood": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "convergence_flag": 999,
                        "omega": np.nan,
                        "alpha_sum": np.nan,
                        "beta_sum": np.nan,
                        "gamma_sum": np.nan,
                        "garch_persistence_alpha_plus_beta": np.nan,
                        "gjr_persistence_approx": np.nan,
                        "nu_student_t": np.nan,
                        "mu": np.nan,
                        "const": np.nan,
                        "ar1": np.nan,
                        "omega_pvalue": np.nan,
                        "alpha1_pvalue": np.nan,
                        "beta1_pvalue": np.nan,
                        "gamma1_pvalue": np.nan,
                        "nu_pvalue": np.nan,
                        "ar1_pvalue": np.nan,
                    }
                )

    comparison = pd.DataFrame(comparison_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    if diagnostics.empty:
        raise RuntimeError("No model was successfully fitted.")

    comparison = comparison.sort_values(["bic", "aic"], na_position="last")
    diagnostics = diagnostics.sort_values("model")

    return comparison, diagnostics, fitted_results


# ============================================================
# 7. MODEL SELECTION
# ============================================================

def choose_best_model(
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    """
    Choose best model using training sample only.

    Selection rule:
    1. Keep converged models.
    2. Prefer models that pass all three residual diagnostics at 5%.
    3. Among them, choose the lowest BIC.
    4. If no model passes all diagnostics, choose lowest BIC among converged models.
    """
    valid = comparison.copy()
    valid = valid[valid["convergence_flag"] == 0]
    valid = valid.dropna(subset=["bic", "aic"])

    if valid.empty:
        raise RuntimeError("No converged model available for selection.")

    diag = diagnostics.copy()
    diag["pass_all_diagnostics_5pct"] = (
        diag["pass_lb_std_resid_5pct"]
        & diag["pass_lb_squared_std_resid_5pct"]
        & diag["pass_arch_lm_5pct"]
    )

    valid = valid.merge(
        diag[["model", "pass_all_diagnostics_5pct"]],
        on="model",
        how="left",
    )

    passing = valid[valid["pass_all_diagnostics_5pct"] == True]  # noqa: E712

    if not passing.empty:
        selected = passing.sort_values(["bic", "aic"]).iloc[0]
        reason = "lowest BIC among models passing all residual diagnostics"
    else:
        selected = valid.sort_values(["bic", "aic"]).iloc[0]
        reason = "lowest BIC among converged models; no model passed all diagnostics"

    best_model_name = str(selected["model"])

    print()
    print("[BEST MODEL - TRAIN SAMPLE]")
    print(f"Selected: {best_model_name}")
    print(f"Reason: {reason}")
    print(f"AIC: {selected['aic']:.3f}")
    print(f"BIC: {selected['bic']:.3f}")
    print()

    return best_model_name


# ============================================================
# 8. FIGURES
# ============================================================

def plot_conditional_volatility(
    result,
    model_name: str,
    output_path: Path,
) -> None:
    """
    Plot estimated conditional volatility from the selected training model.
    """
    cond_vol = pd.Series(result.conditional_volatility)
    cond_vol = cond_vol.replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(12, 5))
    plt.plot(cond_vol.index, cond_vol.values, linewidth=1)
    plt.title(f"Estimated Conditional Volatility - Train - {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Conditional volatility, percent")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_standardized_residual_diagnostics(
    result,
    model_name: str,
    output_path: Path,
) -> None:
    """
    Plot standardized residual diagnostics for the selected training model.
    """
    std_resid = get_standardized_residuals(result)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(std_resid.index, std_resid.values, linewidth=0.8)
    axes[0, 0].axhline(0, linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Standardized Residuals - Train")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Std. residual")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(std_resid.values, bins=40, density=True)
    axes[0, 1].set_title("Histogram of Standardized Residuals - Train")
    axes[0, 1].set_xlabel("Std. residual")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].grid(alpha=0.3)

    plot_acf(std_resid, lags=40, ax=axes[1, 0])
    axes[1, 0].set_title("ACF of Standardized Residuals - Train")

    plot_acf(std_resid**2, lags=40, ax=axes[1, 1])
    axes[1, 1].set_title("ACF of Squared Standardized Residuals - Train")

    fig.suptitle(
        f"Residual Diagnostics - Train - {model_name}",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    fitted_results: dict[str, object],
    best_model_name: str,
) -> None:
    """
    Save tables and figures.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    comparison = comparison.copy()
    diagnostics = diagnostics.copy()

    comparison["is_selected_best_model_train"] = comparison["model"] == best_model_name
    diagnostics["is_selected_best_model_train"] = diagnostics["model"] == best_model_name

    comparison.to_csv(COMPARISON_PATH, index=False)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)

    best_result = fitted_results[best_model_name]

    plot_conditional_volatility(
        best_result,
        best_model_name,
        VOL_FIGURE_PATH,
    )

    plot_standardized_residual_diagnostics(
        best_result,
        best_model_name,
        RESID_FIGURE_PATH,
    )

    print("[SAVED OUTPUTS]")
    print(f"Table:  {COMPARISON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Table:  {DIAGNOSTICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {VOL_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {RESID_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print()


# ============================================================
# 10. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fit GARCH-family models on VN30 training data only.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(TRAIN_DATA_PATH),
        help="Path to training data file.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run full training-sample GARCH model selection pipeline.
    """
    args = parse_args()

    data_path = Path(args.data)

    y_train = load_training_return_series(data_path)

    comparison, diagnostics, fitted_results = fit_all_models(y_train)

    best_model_name = choose_best_model(
        comparison=comparison,
        diagnostics=diagnostics,
    )

    save_outputs(
        comparison=comparison,
        diagnostics=diagnostics,
        fitted_results=fitted_results,
        best_model_name=best_model_name,
    )


if __name__ == "__main__":
    main()