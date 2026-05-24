from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


# ============================================================
# 1. PROJECT PATHS AND CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "VN30_2015_2026.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "vn30_returns_train.csv"
VALID_PATH = PROCESSED_DIR / "vn30_returns_valid.csv"
TEST_PATH = PROCESSED_DIR / "vn30_returns_test.csv"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

SPLIT_SUMMARY_PATH = TABLE_DIR / "00_data_split_summary.csv"
DATA_SUMMARY_PATH = TABLE_DIR / "01_data_summary.csv"
PRELIMINARY_TESTS_PATH = TABLE_DIR / "02_preliminary_tests.csv"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

# For one-step-ahead daily volatility forecasting, gap = 0 is appropriate.
# Use a positive gap only if the target later becomes forward-looking,
# e.g. 5-day or 21-day realized volatility.
GAP_OBS = 0


# ============================================================
# 2. DIRECTORY AND PATH HELPERS
# ============================================================

def ensure_directories() -> None:
    """
    Create necessary project output directories.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def safe_relative_path(path: Path) -> str:
    """
    Print paths relative to project root when possible.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ============================================================
# 3. DATA CLEANING HELPERS
# ============================================================

def read_raw_data(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Read raw VN30 CSV data.

    Expected raw columns:
        Ngày, Lần cuối, Mở, Cao, Thấp, KL, % Thay đổi
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}\n"
            "Please place VN30_2015_2026.csv inside data/raw/ "
            "or pass another path with --raw-path."
        )

    try:
        raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(raw_path, encoding="utf-8")

    raw.columns = [str(col).strip() for col in raw.columns]

    return raw


def clean_price(value: Any) -> float:
    """
    Convert price strings such as '2,022.75' into float.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value in {"", "-", "nan", "None"}:
        return np.nan

    value = value.replace(",", "")

    return pd.to_numeric(value, errors="coerce")


def clean_percent(value: Any) -> float:
    """
    Convert percent strings such as '-0.91%' into float.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value in {"", "-", "nan", "None"}:
        return np.nan

    value = (
        value.replace("%", "")
        .replace(",", "")
        .replace("+", "")
        .replace("−", "-")
        .strip()
    )

    return pd.to_numeric(value, errors="coerce")


def clean_volume(value: Any) -> float:
    """
    Convert volume strings such as:
        '239.35M' -> 239,350,000
        '10.2K'   -> 10,200
        '1.5B'    -> 1,500,000,000
    """
    if pd.isna(value):
        return np.nan

    value = str(value).replace(",", "").strip().upper()

    if value in {"", "-", "NAN", "NONE"}:
        return np.nan

    multiplier = 1.0

    if value.endswith("K"):
        multiplier = 1_000.0
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000.0
        value = value[:-1]
    elif value.endswith("B"):
        multiplier = 1_000_000_000.0
        value = value[:-1]

    numeric_value = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric_value):
        return np.nan

    return float(numeric_value) * multiplier


def standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Vietnamese raw columns into English column names.
    """
    df = raw.copy()
    df.columns = [str(col).strip() for col in df.columns]

    rename_map = {
        "Ngày": "date",
        "Lần cuối": "close",
        "Mở": "open",
        "Cao": "high",
        "Thấp": "low",
        "KL": "volume",
        "% Thay đổi": "raw_change_pct",
    }

    missing_columns = [col for col in rename_map if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Some expected columns are missing from raw data.\n"
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.rename(columns=rename_map)

    return df


def clean_vn30_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw VN30 data and sort by date from oldest to newest.
    """
    df = standardize_columns(raw)

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    for col in ["close", "open", "high", "low"]:
        df[col] = df[col].apply(clean_price)

    df["volume"] = df["volume"].apply(clean_volume)
    df["raw_change_pct"] = df["raw_change_pct"].apply(clean_percent)

    df = (
        df.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    df = df[df["close"] > 0].copy()

    return df


# ============================================================
# 4. RETURN TRANSFORMATION
# ============================================================

def compute_returns(cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Transform VN30 closing index into daily returns.

    Main return variable:
        return_pct = 100 * log(P_t / P_{t-1})

    This scale is appropriate for GARCH modeling because return values
    are in percentage points rather than tiny decimals.
    """
    processed = cleaned.copy()

    processed["log_close"] = np.log(processed["close"])
    processed["log_return"] = processed["log_close"].diff()
    processed["return_pct"] = processed["log_return"] * 100

    processed["simple_return_pct"] = processed["close"].pct_change() * 100

    processed["squared_return"] = processed["return_pct"] ** 2
    processed["abs_return"] = processed["return_pct"].abs()

    processed = processed.dropna().reset_index(drop=True)

    return processed


# ============================================================
# 5. TIME-BASED TRAIN / VALID / TEST SPLIT
# ============================================================

def split_time_series_data(
    processed: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    gap_obs: int = GAP_OBS,
) -> dict[str, pd.DataFrame]:
    """
    Chronologically split processed VN30 return data into train, valid, and test.

    No shuffling is used.

    Parameters
    ----------
    processed:
        Processed VN30 dataframe sorted by date.

    train_ratio:
        Fraction of observations used for training.

    valid_ratio:
        Fraction of observations used for validation.

    gap_obs:
        Number of observations to leave unused between train-valid and valid-test.

        For this project, gap_obs = 0 is appropriate because the forecast task is
        one-step-ahead daily volatility forecasting.

    Returns
    -------
    Dictionary with train, valid, and test dataframes.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")

    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1.")

    if gap_obs < 0:
        raise ValueError("gap_obs must be non-negative.")

    df = processed.sort_values("date").reset_index(drop=True).copy()
    n = len(df)

    if n < 100:
        raise ValueError(
            f"Too few observations after processing: {n}. "
            "Need a larger sample for train/valid/test split."
        )

    n_train = int(np.floor(n * train_ratio))
    n_valid = int(np.floor(n * valid_ratio))

    train_start = 0
    train_end = n_train

    valid_start = train_end + gap_obs
    valid_end = valid_start + n_valid

    test_start = valid_end + gap_obs
    test_end = n

    if valid_start >= valid_end:
        raise ValueError("Split configuration leaves no observations for validation.")

    if test_start >= test_end:
        raise ValueError(
            "Split configuration leaves no observations for the test set. "
            "Reduce train_ratio, valid_ratio, or gap_obs."
        )

    train = df.iloc[train_start:train_end].copy()
    valid = df.iloc[valid_start:valid_end].copy()
    test = df.iloc[test_start:test_end].copy()

    train["split"] = "train"
    valid["split"] = "valid"
    test["split"] = "test"

    return {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def make_split_summary(
    splits: dict[str, pd.DataFrame],
    gap_obs: int = GAP_OBS,
) -> pd.DataFrame:
    """
    Create a compact split summary table for reporting.
    """
    rows: list[dict[str, Any]] = []

    for split_name in ["train", "valid", "test"]:
        split_df = splits[split_name]

        rows.append(
            {
                "split": split_name,
                "observations": len(split_df),
                "start_date": split_df["date"].min(),
                "end_date": split_df["date"].max(),
                "mean_return_pct": split_df["return_pct"].mean(),
                "std_return_pct": split_df["return_pct"].std(),
                "min_return_pct": split_df["return_pct"].min(),
                "max_return_pct": split_df["return_pct"].max(),
            }
        )

    summary = pd.DataFrame(rows)
    summary["gap_between_splits_obs"] = gap_obs

    return summary


def save_processed_splits(
    splits: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """
    Save train, validation, and test processed datasets.
    """
    split_paths = {
        "train": TRAIN_PATH,
        "valid": VALID_PATH,
        "test": TEST_PATH,
    }

    for split_name, path in split_paths.items():
        splits[split_name].to_csv(path, index=False)

    return split_paths


# ============================================================
# 6. SUMMARY TABLES
# ============================================================

def make_data_summary_table(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    processed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one compact table summarizing data quality and return distribution.
    """
    r = processed["return_pct"].dropna()

    summary = pd.DataFrame(
        {
            "Metric": [
                "Raw rows",
                "Raw columns",
                "Duplicated raw rows",
                "Total raw missing values",
                "Cleaned observations",
                "Processed return observations",
                "Start date",
                "End date",
                "Duplicated dates after cleaning",
                "Missing close values after cleaning",
                "Minimum close",
                "Maximum close",
                "Mean return (%)",
                "Median return (%)",
                "Std. dev. return (%)",
                "Minimum return (%)",
                "Maximum return (%)",
                "Skewness",
                "Excess kurtosis",
                "Kurtosis",
            ],
            "Value": [
                raw.shape[0],
                raw.shape[1],
                raw.duplicated().sum(),
                raw.isna().sum().sum(),
                len(cleaned),
                len(processed),
                cleaned["date"].min(),
                cleaned["date"].max(),
                cleaned["date"].duplicated().sum(),
                cleaned["close"].isna().sum(),
                cleaned["close"].min(),
                cleaned["close"].max(),
                r.mean(),
                r.median(),
                r.std(),
                r.min(),
                r.max(),
                r.skew(),
                r.kurtosis(),
                r.kurtosis() + 3,
            ],
        }
    )

    return summary


def save_report_tables(
    data_summary: pd.DataFrame,
    preliminary_tests: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> dict[str, Path]:
    """
    Save compact report tables.
    """
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False)
    data_summary.to_csv(DATA_SUMMARY_PATH, index=False)
    preliminary_tests.to_csv(PRELIMINARY_TESTS_PATH, index=False)

    return {
        "split_summary": SPLIT_SUMMARY_PATH,
        "data_summary": DATA_SUMMARY_PATH,
        "preliminary_tests": PRELIMINARY_TESTS_PATH,
    }


# ============================================================
# 7. STATISTICAL TESTS
# ============================================================

def run_adf_test(
    series: pd.Series,
    name: str,
    regression: str = "c",
) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller test.

    regression:
    - 'c'  : constant only
    - 'ct' : constant and trend
    """
    x = pd.Series(series).dropna()

    if len(x) < 30:
        raise ValueError(f"Series '{name}' is too short for ADF test.")

    result = adfuller(x, regression=regression, autolag="AIC")

    table = pd.DataFrame(
        {
            "Series": [name],
            "ADF statistic": [result[0]],
            "p-value": [result[1]],
            "Used lag": [result[2]],
            "Observations": [result[3]],
            "1% critical value": [result[4]["1%"]],
            "5% critical value": [result[4]["5%"]],
            "10% critical value": [result[4]["10%"]],
        }
    )

    return table


def run_adf_tests(processed_train: pd.DataFrame) -> pd.DataFrame:
    """
    Run ADF test on VN30 closing level and VN30 return using training data only.
    """
    adf_close = run_adf_test(
        processed_train["close"],
        name="VN30 close",
        regression="ct",
    )

    adf_return = run_adf_test(
        processed_train["return_pct"],
        name="VN30 return (%)",
        regression="c",
    )

    adf_results = pd.concat([adf_close, adf_return], ignore_index=True)

    return adf_results


def run_ljung_box_tests(
    processed_train: pd.DataFrame,
    lags: list[int] | tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    """
    Run Ljung-Box tests on returns and squared returns using training data only.
    """
    returns = processed_train["return_pct"].dropna()
    squared_returns = processed_train["squared_return"].dropna()

    return_lb = acorr_ljungbox(
        returns,
        lags=list(lags),
        return_df=True,
    )
    return_lb.insert(0, "Lag", return_lb.index)
    return_lb.insert(0, "Series", "Return")

    squared_lb = acorr_ljungbox(
        squared_returns,
        lags=list(lags),
        return_df=True,
    )
    squared_lb.insert(0, "Lag", squared_lb.index)
    squared_lb.insert(0, "Series", "Squared return")

    ljung_results = pd.concat([return_lb, squared_lb], ignore_index=True)
    ljung_results = ljung_results[["Series", "Lag", "lb_stat", "lb_pvalue"]]

    return ljung_results


def run_arch_lm_test(
    processed_train: pd.DataFrame,
    nlags: int = 10,
) -> pd.DataFrame:
    """
    Run ARCH-LM test on demeaned VN30 returns using training data only.

    Null hypothesis:
        No ARCH effects.

    If p-value is small, ARCH/GARCH modeling is justified.
    """
    returns = processed_train["return_pct"].dropna()
    demeaned_returns = returns - returns.mean()

    result = het_arch(demeaned_returns, nlags=nlags)

    table = pd.DataFrame(
        {
            "Test": ["ARCH-LM"],
            "Lags": [nlags],
            "LM statistic": [result[0]],
            "LM p-value": [result[1]],
            "F statistic": [result[2]],
            "F p-value": [result[3]],
        }
    )

    return table


def make_preliminary_test_summary(
    adf_results: pd.DataFrame,
    ljung_results: pd.DataFrame,
    arch_lm_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine important preliminary tests into one compact summary table.
    """
    adf_close = adf_results.loc[
        adf_results["Series"] == "VN30 close"
    ].iloc[0]

    adf_return = adf_results.loc[
        adf_results["Series"] == "VN30 return (%)"
    ].iloc[0]

    lb_return_20 = ljung_results.loc[
        (ljung_results["Series"] == "Return")
        & (ljung_results["Lag"] == 20)
    ].iloc[0]

    lb_squared_20 = ljung_results.loc[
        (ljung_results["Series"] == "Squared return")
        & (ljung_results["Lag"] == 20)
    ].iloc[0]

    arch_lm = arch_lm_table.iloc[0]

    summary = pd.DataFrame(
        {
            "Test": [
                "ADF test on VN30 close",
                "ADF test on VN30 return",
                "Ljung-Box on return, lag 20",
                "Ljung-Box on squared return, lag 20",
                "ARCH-LM test, lag 10",
            ],
            "Sample": [
                "Train only",
                "Train only",
                "Train only",
                "Train only",
                "Train only",
            ],
            "Statistic": [
                adf_close["ADF statistic"],
                adf_return["ADF statistic"],
                lb_return_20["lb_stat"],
                lb_squared_20["lb_stat"],
                arch_lm["LM statistic"],
            ],
            "p-value": [
                adf_close["p-value"],
                adf_return["p-value"],
                lb_return_20["lb_pvalue"],
                lb_squared_20["lb_pvalue"],
                arch_lm["LM p-value"],
            ],
        }
    )

    return summary


# ============================================================
# 8. REPORT-STYLE PLOTS
# ============================================================

def _save_or_show(fig: plt.Figure, path: Path, show_plots: bool) -> None:
    """
    Save a matplotlib figure and optionally show it.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def make_eda_overview_plot(
    processed: pd.DataFrame,
    show_plots: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """
    Create one compact EDA overview figure using the full processed sample.

    This figure is descriptive only. Formal statistical tests and model selection
    should use the training set only.
    """
    path = FIGURE_DIR / "01_eda_overview.png"

    output = processed.copy()
    output["rolling_vol_21"] = output["return_pct"].rolling(window=21).std()
    output["rolling_vol_63"] = output["return_pct"].rolling(window=63).std()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(output["date"], output["close"], linewidth=1)
    axes[0, 0].set_title("VN30 Closing Price")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Close")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(output["date"], output["return_pct"], linewidth=0.8)
    axes[0, 1].axhline(0, linewidth=1)
    axes[0, 1].set_title("VN30 Daily Log Return (%)")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Return (%)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(output["return_pct"].dropna(), bins=60, density=True)
    axes[1, 0].set_title("Distribution of Daily Log Return (%)")
    axes[1, 0].set_xlabel("Return (%)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(
        output["date"],
        output["rolling_vol_21"],
        linewidth=1,
        label="21-day rolling volatility",
    )
    axes[1, 1].plot(
        output["date"],
        output["rolling_vol_63"],
        linewidth=1,
        label="63-day rolling volatility",
    )
    axes[1, 1].set_title("Rolling Volatility")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Rolling std. dev. of return (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("VN30 Data and Return Overview", fontsize=16, fontweight="bold")

    _save_or_show(fig, path, show_plots)

    return output, path


def make_volatility_diagnostics_plot(
    processed_train: pd.DataFrame,
    show_plots: bool = False,
) -> Path:
    """
    Create one compact volatility diagnostics figure using the training sample only.

    The figure includes:
    1. ACF of returns
    2. PACF of returns
    3. ACF of squared returns
    4. ACF of absolute returns
    """
    path = FIGURE_DIR / "02_volatility_diagnostics_train.png"

    returns = processed_train["return_pct"].dropna()

    if len(returns) < 50:
        raise ValueError("Training sample is too short for ACF/PACF diagnostics.")

    max_lags = min(40, len(returns) // 2 - 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_acf(returns, lags=max_lags, ax=axes[0, 0])
    axes[0, 0].set_title("ACF of VN30 Return - Train")

    plot_pacf(returns, lags=max_lags, ax=axes[0, 1], method="ywm")
    axes[0, 1].set_title("PACF of VN30 Return - Train")

    plot_acf(processed_train["squared_return"].dropna(), lags=max_lags, ax=axes[1, 0])
    axes[1, 0].set_title("ACF of Squared Return - Train")

    plot_acf(processed_train["abs_return"].dropna(), lags=max_lags, ax=axes[1, 1])
    axes[1, 1].set_title("ACF of Absolute Return - Train")

    fig.suptitle(
        "VN30 Return Dependence and Volatility Diagnostics - Training Sample",
        fontsize=16,
        fontweight="bold",
    )

    _save_or_show(fig, path, show_plots)

    return path


def make_report_plots(
    processed: pd.DataFrame,
    processed_train: pd.DataFrame,
    show_plots: bool = False,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """
    Create compact report-style figures.

    - EDA overview uses full sample for descriptive visualization.
    - Volatility diagnostics use training sample only.
    """
    processed_with_rolling, eda_path = make_eda_overview_plot(
        processed=processed,
        show_plots=show_plots,
    )

    diagnostics_path = make_volatility_diagnostics_plot(
        processed_train=processed_train,
        show_plots=show_plots,
    )

    figure_paths = {
        "eda_overview": eda_path,
        "volatility_diagnostics_train": diagnostics_path,
    }

    return processed_with_rolling, figure_paths


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def run_data_eda(
    raw_path: Path = RAW_PATH,
    show_plots: bool = False,
    save_outputs: bool = True,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    gap_obs: int = GAP_OBS,
) -> dict[str, Any]:
    """
    Run the full data preparation, EDA, split, and preliminary testing pipeline.

    Important design choices:
    - Data are split chronologically into train / valid / test.
    - Preliminary statistical tests are run on the training sample only.
    - Processed data are saved as exactly three files:
        data/processed/vn30_returns_train.csv
        data/processed/vn30_returns_valid.csv
        data/processed/vn30_returns_test.csv
    """
    ensure_directories()

    raw = read_raw_data(raw_path)
    cleaned = clean_vn30_data(raw)
    processed = compute_returns(cleaned)

    splits = split_time_series_data(
        processed=processed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        gap_obs=gap_obs,
    )

    train = splits["train"]

    adf_results = run_adf_tests(train)
    ljung_results = run_ljung_box_tests(train, lags=(10, 20))
    arch_lm_table = run_arch_lm_test(train, nlags=10)

    preliminary_tests = make_preliminary_test_summary(
        adf_results=adf_results,
        ljung_results=ljung_results,
        arch_lm_table=arch_lm_table,
    )

    processed_with_rolling, figure_paths = make_report_plots(
        processed=processed,
        processed_train=train,
        show_plots=show_plots,
    )

    data_summary = make_data_summary_table(
        raw=raw,
        cleaned=cleaned,
        processed=processed,
    )

    split_summary = make_split_summary(
        splits=splits,
        gap_obs=gap_obs,
    )

    table_paths: dict[str, Path] = {}
    split_paths: dict[str, Path] = {}

    if save_outputs:
        split_paths = save_processed_splits(splits)

        table_paths = save_report_tables(
            data_summary=data_summary,
            preliminary_tests=preliminary_tests,
            split_summary=split_summary,
        )

    results = {
        "raw": raw,
        "cleaned": cleaned,
        "processed": processed,
        "processed_with_rolling": processed_with_rolling,
        "splits": splits,
        "train": splits["train"],
        "valid": splits["valid"],
        "test": splits["test"],
        "split_summary": split_summary,
        "data_summary": data_summary,
        "adf_results": adf_results,
        "ljung_results": ljung_results,
        "arch_lm_table": arch_lm_table,
        "preliminary_tests": preliminary_tests,
        "figure_paths": figure_paths,
        "table_paths": table_paths,
        "split_paths": split_paths,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": 1 - train_ratio - valid_ratio,
        "gap_obs": gap_obs,
    }

    return results


# ============================================================
# 10. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse terminal arguments.

    Recommended:
        uv run python src/vn30_garch/data_eda.py
    """
    parser = argparse.ArgumentParser(
        description="Clean VN30 data, compute returns, split train/valid/test, and run EDA tests.",
    )

    parser.add_argument(
        "--raw-path",
        type=str,
        default=str(RAW_PATH),
        help="Path to raw VN30 CSV file.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Chronological training ratio.",
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=VALID_RATIO,
        help="Chronological validation ratio.",
    )

    parser.add_argument(
        "--gap-obs",
        type=int,
        default=GAP_OBS,
        help="Number of observations to leave unused between splits.",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively instead of only saving them.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run from terminal:

        uv run python src/vn30_garch/data_eda.py
    """
    args = parse_args()

    results = run_data_eda(
        raw_path=Path(args.raw_path),
        show_plots=args.show_plots,
        save_outputs=True,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        gap_obs=args.gap_obs,
    )

    print("\nData EDA pipeline completed.")

    print("\nProcessed split files:")
    for split_name, path in results["split_paths"].items():
        print(f"  {split_name:5s}: {safe_relative_path(path)}")

    print("\nSaved tables:")
    for table_name, path in results["table_paths"].items():
        print(f"  {table_name:20s}: {safe_relative_path(path)}")

    print("\nSaved figures:")
    for figure_name, path in results["figure_paths"].items():
        print(f"  {figure_name:30s}: {safe_relative_path(path)}")

    print("\nSplit summary:")
    print(results["split_summary"].to_string(index=False))

    print(f"\nGap between splits: {results['gap_obs']} trading observations")

    print("\nData summary:")
    print(results["data_summary"].to_string(index=False))

    print("\nPreliminary test summary:")
    print(results["preliminary_tests"].to_string(index=False))


if __name__ == "__main__":
    main()