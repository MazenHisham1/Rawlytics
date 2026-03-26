import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def data_info(data):
    return {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "dtypes": data.dtypes.astype(str).to_dict()
    }

def data_desc(data):
    return data.describe().to_dict()

def NULL_Percentage(data):
    return (data.isna().mean() * 100).to_dict()

def Number_columns(data):
    return data.shape[1]

def Number_rows(data):
    return data.shape[0]

def Number_duplicates(data):
    return int(data.duplicated().sum())

def col_type(data, col):
    return str(data[col].dtype)

def Number_Uniques(data, col):
    return int(data[col].nunique())

def calculate_skewness(series: pd.Series) -> float:
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("Skewness requires numeric data")
    series = series.dropna()
    n = len(series)
    if n < 3:
        return 0.0
    mean = series.mean()
    std = series.std(ddof=1)
    skewness = (n / ((n - 1) * (n - 2))) * np.sum(((series - mean) / std) ** 3)
    return skewness

def skew_type(skew):
    if skew < -0.5:
        return "left-skewed"
    elif skew > 0.5:
        return "right-skewed"
    else:
        return "approximately symmetric"

def missing_rate(df, column):
    return df[column].isnull().mean()

def skewness(series):
    return series.skew()

def has_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).any()

def strong_correlations(df, target_col, threshold=0.4):
    numeric_cols = df.select_dtypes(include='number').columns
    corr = df[numeric_cols].corr()[target_col].drop(target_col)
    return corr[abs(corr) >= threshold]

def recommend_imputation_method(df, column):
    report = {}
    miss_rate = missing_rate(df, column)
    report["missing_rate"] = float(round(miss_rate, 3))
    series = df[column].dropna()
    report["skewness"] = float(round(series.skew(), 3))
    report["has_outliers"] = bool(has_outliers(series))
    report["dataset_size"] = int(len(df))
    correlations = strong_correlations(df, column)
    report["strong_correlations"] = {str(k): float(v) for k, v in correlations.to_dict().items()}
    num_features = len(df.select_dtypes(include="number").columns)

    if miss_rate < 0.05:
        method = "Median Imputation" if report["has_outliers"] else "Mean Imputation"
    elif miss_rate < 0.3:
        if len(correlations) == 0:
            method = "Median Imputation"
        else:
            if len(correlations) == 1:
                method = "Regression Imputation"
            elif num_features >= 6 and report["dataset_size"] >= 50:
                method = "Iterative Imputation (MICE)"
            else:
                method = "KNN Imputation"
    else:
        if len(correlations) == 0:
            method = "Median Imputation (with caution)"
        else:
            if num_features >= 6 and report["dataset_size"] >= 100:
                method = "Iterative Imputation (MICE)"
            else:
                method = "KNN Imputation"

    if report["dataset_size"] < 50 and method in {"KNN Imputation", "Iterative Imputation (MICE)", "Regression Imputation"}:
        method = "Median Imputation (small dataset fallback)"

    report["recommended_method"] = str(method)
    return report


# =========================
# 1. Statistical depth
# =========================

def calculate_kurtosis(series: pd.Series) -> dict:
    """Return excess kurtosis and its interpretation."""
    series = series.dropna()
    n = len(series)
    if n < 4:
        return {"kurtosis": None, "type": "insufficient data"}
    kurt = float(series.kurtosis())  # pandas uses excess kurtosis (Fisher)
    if kurt > 1.0:
        kind = "leptokurtic (heavy tails / sharp peak)"
    elif kurt < -1.0:
        kind = "platykurtic (thin tails / flat peak)"
    else:
        kind = "mesokurtic (near-normal)"
    return {"kurtosis": round(kurt, 4), "type": kind}


def normality_test(series: pd.Series) -> dict:
    """
    Run Shapiro-Wilk for n<=5000, Kolmogorov-Smirnov otherwise.
    Returns test name, statistic, p-value, and a plain-English verdict.
    """
    series = series.dropna()
    n = len(series)
    if n < 3:
        return {"error": "Need at least 3 non-null values"}

    if n <= 5000:
        stat, p = scipy_stats.shapiro(series)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = scipy_stats.kstest(series, "norm",
                                     args=(series.mean(), series.std(ddof=1)))
        test_name = "Kolmogorov-Smirnov"

    alpha = 0.05
    verdict = "likely normal" if p > alpha else "not normal"
    return {
        "test": test_name,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "alpha": alpha,
        "verdict": verdict,
        "n": n
    }


def value_entropy(series: pd.Series) -> dict:
    """Shannon entropy for a categorical or low-cardinality column."""
    counts = series.dropna().value_counts(normalize=True)
    ent = float(-np.sum(counts * np.log2(counts + 1e-12)))
    max_ent = float(np.log2(len(counts))) if len(counts) > 1 else 1.0
    normalized = round(ent / max_ent, 4) if max_ent > 0 else 0.0
    return {
        "entropy_bits": round(ent, 4),
        "max_possible_bits": round(max_ent, 4),
        "normalized_entropy": normalized,
        "interpretation": (
            "highly concentrated (low diversity)" if normalized < 0.3
            else "moderately diverse" if normalized < 0.7
            else "highly diverse / near-uniform"
        )
    }


def correlation_matrix(df: pd.DataFrame) -> dict:
    """
    Pearson correlation for numeric pairs.
    Cramér's V for categorical pairs.
    Returns separate matrices.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    pearson = {}
    if len(numeric_cols) >= 2:
        pearson = df[numeric_cols].corr(method="pearson").round(4).to_dict()

    def cramers_v(x, y):
        ct = pd.crosstab(x, y)
        chi2 = scipy_stats.chi2_contingency(ct, correction=False)[0]
        n = ct.values.sum()
        phi2 = chi2 / n
        r, k = ct.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - (r - 1) ** 2 / (n - 1)
        kcorr = k - (k - 1) ** 2 / (n - 1)
        denom = min(kcorr - 1, rcorr - 1)
        if denom <= 0:
            return 0.0
        return round(float(np.sqrt(phi2corr / denom)), 4)

    cramers = {}
    if len(cat_cols) >= 2:
        for c1 in cat_cols:
            cramers[c1] = {}
            for c2 in cat_cols:
                if c1 == c2:
                    cramers[c1][c2] = 1.0
                else:
                    try:
                        cramers[c1][c2] = cramers_v(
                            df[c1].dropna(), df[c2].dropna()
                        )
                    except Exception:
                        cramers[c1][c2] = None

    return {
        "pearson": pearson,
        "cramers_v": cramers,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols
    }


# =========================
# 2. Outlier detection
# =========================

def detect_outliers(df: pd.DataFrame, col: str, method: str = "iqr") -> dict:
    """
    Detect outliers using IQR or Z-score method.
    Returns flagged row indices, values, and summary stats.
    """
    series = df[col].dropna()
    n_total = int(len(df))
    n_valid = int(len(series))

    if method == "zscore":
        z = np.abs(scipy_stats.zscore(series))
        mask = z > 3
        threshold_desc = "|z| > 3"
        lower_bound = None
        upper_bound = None
    else:  # iqr
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = float(q1 - 1.5 * iqr)
        upper_bound = float(q3 + 1.5 * iqr)
        mask = (series < lower_bound) | (series > upper_bound)
        threshold_desc = f"outside [{round(lower_bound, 4)}, {round(upper_bound, 4)}]"

    outlier_series = series[mask]
    outlier_indices = [int(i) for i in outlier_series.index.tolist()]
    outlier_values = [float(v) for v in outlier_series.values.tolist()]

    return {
        "method": method,
        "threshold": threshold_desc,
        "lower_fence": lower_bound,
        "upper_fence": upper_bound,
        "n_outliers": len(outlier_indices),
        "pct_outliers": round(len(outlier_indices) / n_valid * 100, 2) if n_valid > 0 else 0,
        "outlier_indices": outlier_indices[:100],  # cap at 100 for response size
        "outlier_values": outlier_values[:100],
        "n_total": n_total,
        "n_valid": n_valid
    }


# =========================
# 3. Imputation — apply it
# =========================

def apply_imputation(df: pd.DataFrame, col: str, method: str) -> dict:
    """
    Apply the specified imputation method and return the updated column values.
    Supported methods: mean, median, mode, constant, ffill, bfill.
    Returns count of filled values and the new series as a list.
    """
    series = df[col].copy()
    n_missing_before = int(series.isna().sum())

    method = method.lower()
    if method == "mean":
        fill_value = series.mean()
        series = series.fillna(fill_value)
    elif method == "median":
        fill_value = series.median()
        series = series.fillna(fill_value)
    elif method == "mode":
        fill_value = series.mode().iloc[0] if not series.mode().empty else None
        series = series.fillna(fill_value)
    elif method == "ffill":
        series = series.ffill()
        fill_value = "forward fill"
    elif method == "bfill":
        series = series.bfill()
        fill_value = "backward fill"
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: mean, median, mode, ffill, bfill")

    n_missing_after = int(series.isna().sum())
    n_filled = n_missing_before - n_missing_after

    return {
        "column": col,
        "method_applied": method,
        "fill_value": str(fill_value) if not isinstance(fill_value, str) else fill_value,
        "n_missing_before": n_missing_before,
        "n_missing_after": n_missing_after,
        "n_filled": n_filled,
        "values": series.tolist()
    }


# =========================
# 4. Data type inference
# =========================

def suggest_dtypes(df: pd.DataFrame) -> dict:
    """
    Heuristically suggest better dtypes for each column.
    Detects: dates, booleans, IDs, categories, and miscast numerics.
    """
    suggestions = {}
    n = len(df)

    for col in df.columns:
        series = df[col]
        current = str(series.dtype)
        suggestion = None
        reason = None

        if pd.api.types.is_object_dtype(series):
            sample = series.dropna().head(200)

            # Try datetime
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                suggestion = "datetime64"
                reason = "values parse successfully as dates"
            except Exception:
                pass

            # Boolean check
            if suggestion is None:
                uniq = set(series.dropna().str.lower().unique()) if series.dtype == object else set()
                bool_sets = [{"true", "false"}, {"yes", "no"}, {"1", "0"}, {"t", "f"}, {"y", "n"}]
                if any(uniq <= b for b in bool_sets):
                    suggestion = "bool"
                    reason = "only two distinct boolean-like values"

            # Category check
            if suggestion is None:
                nuniq = series.nunique()
                if nuniq / n < 0.1 and nuniq <= 50:
                    suggestion = "category"
                    reason = f"low cardinality ({nuniq} unique values, {round(nuniq/n*100,1)}% of rows)"

            # Try numeric
            if suggestion is None:
                try:
                    pd.to_numeric(series.dropna().head(200))
                    suggestion = "float64 or int64"
                    reason = "values are numeric but stored as object"
                except Exception:
                    pass

        elif pd.api.types.is_float_dtype(series):
            # ID-like: all unique, no decimals
            if series.nunique() == n and (series.dropna() % 1 == 0).all():
                suggestion = "int64 (ID column)"
                reason = "all unique, integer values — likely an ID"
            # Could be int
            elif (series.dropna() % 1 == 0).all():
                suggestion = "int64"
                reason = "all values are whole numbers stored as float"

        elif pd.api.types.is_integer_dtype(series):
            if series.nunique() == n:
                suggestion = "keep as int64 (ID column)"
                reason = "all unique — likely an ID"

        suggestions[col] = {
            "current_dtype": current,
            "suggested_dtype": suggestion if suggestion else "keep as-is",
            "reason": reason if reason else "dtype looks appropriate"
        }

    return suggestions


# =========================
# 5. ML readiness
# =========================

def encoding_suggestions(df: pd.DataFrame) -> dict:
    """
    For each categorical column, recommend an encoding strategy
    based on cardinality relative to dataset size.
    """
    n = len(df)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    result = {}

    for col in cat_cols:
        nuniq = df[col].nunique()
        ratio = nuniq / n if n > 0 else 0

        if nuniq == 2:
            method = "label encoding (binary)"
            reason = "only 2 unique values"
        elif nuniq <= 10:
            method = "one-hot encoding"
            reason = f"low cardinality ({nuniq} unique values)"
        elif nuniq <= 50 or ratio < 0.05:
            method = "one-hot encoding (with care) or ordinal encoding"
            reason = f"medium cardinality ({nuniq} unique values, {round(ratio*100,1)}% of rows)"
        else:
            method = "target encoding or hashing"
            reason = f"high cardinality ({nuniq} unique values, {round(ratio*100,1)}% of rows) — one-hot would create too many features"

        result[col] = {
            "n_unique": int(nuniq),
            "cardinality_ratio": round(ratio, 4),
            "recommended_encoding": method,
            "reason": reason
        }

    return result


def feature_importance_proxy(df: pd.DataFrame, target_col: str) -> dict:
    """
    Compute mutual information scores between all features and target column.
    Works for both classification (categorical target) and regression (numeric target).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    df_clean = df[feature_cols + [target_col]].dropna()

    if len(df_clean) < 10:
        return {"error": "Not enough rows after dropping nulls (need >= 10)"}

    # Encode categoricals for MI computation
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col]

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes

    is_classification = not pd.api.types.is_numeric_dtype(y)
    if is_classification:
        y = y.astype("category").cat.codes
        scores = mutual_info_classif(X, y, random_state=42)
        task = "classification"
    else:
        scores = mutual_info_regression(X, y, random_state=42)
        task = "regression"

    ranked = sorted(
        zip(feature_cols, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "target": target_col,
        "task_type": task,
        "ranking": [
            {
                "feature": feat,
                "mutual_info_score": round(score, 4),
                "importance": (
                    "high" if score > 0.3
                    else "medium" if score > 0.05
                    else "low"
                )
            }
            for feat, score in ranked
        ]
    }


def class_balance(df: pd.DataFrame, col: str) -> dict:
    """
    For a classification target column: show class counts,
    ratios, and imbalance flag if any class < 20%.
    """
    counts = df[col].value_counts()
    total = int(counts.sum())
    ratios = (counts / total).round(4)
    min_ratio = float(ratios.min())
    is_imbalanced = min_ratio < 0.20

    return {
        "column": col,
        "n_classes": int(len(counts)),
        "total_samples": total,
        "class_distribution": [
            {
                "class": str(k),
                "count": int(v),
                "ratio": float(round(ratios[k], 4)),
                "pct": float(round(ratios[k] * 100, 2))
            }
            for k, v in counts.items()
        ],
        "is_imbalanced": is_imbalanced,
        "minority_ratio": min_ratio,
        "imbalance_warning": (
            f"Minority class ratio is {round(min_ratio*100,1)}% — consider oversampling (SMOTE), "
            f"undersampling, or class_weight='balanced'"
            if is_imbalanced else None
        )
    }


# =========================
# 6. Full auto EDA report
# =========================

def full_eda_report(df: pd.DataFrame) -> dict:
    """
    One-call comprehensive EDA report combining all analyses.
    """
    report = {}

    # Overview
    report["overview"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": int(df.shape[0] * df.shape[1]),
        "duplicates": Number_duplicates(df),
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }

    # Per-column analysis
    report["columns"] = {}
    for col in df.columns:
        col_report = {}
        series = df[col]
        col_report["dtype"] = str(series.dtype)
        col_report["null_count"] = int(series.isna().sum())
        col_report["null_pct"] = round(float(series.isna().mean() * 100), 2)
        col_report["n_unique"] = int(series.nunique())

        if pd.api.types.is_numeric_dtype(series):
            s = series.dropna()
            col_report["mean"] = round(float(s.mean()), 4) if len(s) else None
            col_report["median"] = round(float(s.median()), 4) if len(s) else None
            col_report["std"] = round(float(s.std()), 4) if len(s) else None
            col_report["min"] = round(float(s.min()), 4) if len(s) else None
            col_report["max"] = round(float(s.max()), 4) if len(s) else None
            col_report["skewness"] = round(float(calculate_skewness(s)), 4) if len(s) >= 3 else None
            col_report["skew_type"] = skew_type(col_report["skewness"]) if col_report["skewness"] is not None else None
            col_report["kurtosis"] = calculate_kurtosis(s)
            col_report["outliers_iqr"] = detect_outliers(df, col, method="iqr")
            col_report["normality"] = normality_test(s)
            col_report["imputation_recommendation"] = recommend_imputation_method(df, col)
        else:
            col_report["top_values"] = series.value_counts().head(5).to_dict()
            col_report["entropy"] = value_entropy(series)

        report["columns"][col] = col_report

    # Correlation
    report["correlation"] = correlation_matrix(df)

    # Dtype suggestions
    report["dtype_suggestions"] = suggest_dtypes(df)

    # Encoding suggestions
    report["encoding_suggestions"] = encoding_suggestions(df)

    return report
