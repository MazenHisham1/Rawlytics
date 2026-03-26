from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import asyncio
import io
from uuid import uuid4

from functions import (
    # Existing
    Number_rows, Number_columns, data_info, data_desc,
    NULL_Percentage, Number_duplicates, col_type, Number_Uniques,
    calculate_skewness, skew_type, recommend_imputation_method,
    # New — statistical depth
    calculate_kurtosis, normality_test, value_entropy, correlation_matrix,
    # New — outlier detection
    detect_outliers,
    # New — imputation apply
    apply_imputation,
    # New — dtype inference
    suggest_dtypes,
    # New — ML readiness
    encoding_suggestions, feature_importance_proxy, class_balance,
    # New — auto report
    full_eda_report,
)


# =========================
# App setup
# =========================
app = FastAPI(
    title="EDA API",
    description="Comprehensive Exploratory Data Analysis API with statistical depth, ML readiness checks, and auto-reporting.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Session store
# (replaces global df — supports multiple concurrent users)
# =========================
sessions: dict[str, pd.DataFrame] = {}

MAX_FILE_SIZE_MB = 50


# =========================
# Helpers
# =========================
def get_df(session_id: str) -> pd.DataFrame:
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. POST a CSV to /upload first."
        )
    return sessions[session_id]


def require_col(df: pd.DataFrame, col: str):
    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")


def require_numeric(df: pd.DataFrame, col: str):
    require_col(df, col)
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise HTTPException(status_code=400, detail=f"Column '{col}' must be numeric.")


# =========================
# Pydantic request models
# =========================
class ImputeRequest(BaseModel):
    method: str  # mean | median | mode | ffill | bfill


# =========================
# Root
# =========================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_sessions": len(sessions)
    }


@app.post("/ping")
def ping():
    return {"status": "server is running"}


# =========================
# Upload
# =========================
@app.post("/upload", summary="Upload a CSV file and get a session_id")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV. Returns a `session_id` to use in all subsequent requests.
    Max file size: 50 MB.
    """
    contents = await file.read()

    # Size guard
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({round(size_mb, 1)} MB). Max allowed: {MAX_FILE_SIZE_MB} MB."
        )

    try:
        df = await asyncio.to_thread(pd.read_csv, io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    session_id = str(uuid4())
    sessions[session_id] = df

    return {
        "session_id": session_id,
        "filename": file.filename,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "size_mb": round(size_mb, 3)
    }


@app.delete("/session/{session_id}", summary="Delete a session and free memory")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    return {"deleted": session_id}


# =========================
# Basic data endpoints
# =========================
@app.post("/shape")
def shape(session_id: str = Query(...)):
    df = get_df(session_id)
    return {"rows": Number_rows(df), "columns": Number_columns(df)}


@app.post("/info")
def info(session_id: str = Query(...)):
    df = get_df(session_id)
    return data_info(df)


@app.post("/describe")
def describe(session_id: str = Query(...)):
    df = get_df(session_id)
    return data_desc(df)


@app.post("/nulls")
def nulls(session_id: str = Query(...)):
    df = get_df(session_id)
    return NULL_Percentage(df)


@app.post("/duplicates")
def duplicates(session_id: str = Query(...)):
    df = get_df(session_id)
    return {"duplicates": Number_duplicates(df)}


@app.post("/column/type/{col}")
def column_type(col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    require_col(df, col)
    return {"column": col, "type": col_type(df, col)}


@app.post("/column/unique/{col}")
def column_unique(col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    require_col(df, col)
    return {"column": col, "unique": Number_Uniques(df, col)}


# =========================
# Chart endpoints
# =========================
@app.post("/chart/distribution/{col}")
def distribution(col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    require_col(df, col)
    dist = df[col].value_counts()
    return [{"label": str(k), "value": int(v)} for k, v in dist.items()]


@app.post("/chart/bar/{col}")
def bar_chart(col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    require_col(df, col)
    counts = df[col].value_counts()
    return [{"label": str(k), "value": int(v)} for k, v in counts.items()]


@app.post("/chart/pie/{col}")
def pie_chart(col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    require_col(df, col)
    counts = df[col].value_counts()
    return [{"label": str(k), "value": int(v)} for k, v in counts.items()]


@app.post("/chart/hist/{col}")
def histogram(
    col: str,
    session_id: str = Query(...),
    bins: int = Query(10, ge=1, le=200)
):
    df = get_df(session_id)
    require_numeric(df, col)
    counts = pd.cut(df[col], bins=bins, include_lowest=True).value_counts().sort_index()
    return [{"label": str(interval), "value": int(count)} for interval, count in counts.items()]


@app.post("/chart/scatter/{x_col}/{y_col}")
def scatter_plot(x_col: str, y_col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    for col in [x_col, y_col]:
        require_numeric(df, col)
    data = df[[x_col, y_col]].dropna()
    return [{"x": float(row[x_col]), "y": float(row[y_col])} for _, row in data.iterrows()]


@app.post("/chart/boxplot/{col}")
def box_plot(col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    require_numeric(df, col)
    series = df[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = series[(series < lower_fence) | (series > upper_fence)].tolist()
    return {
        "min": float(series.min()),
        "q1": float(q1),
        "median": float(series.median()),
        "q3": float(q3),
        "max": float(series.max()),
        "outliers": [float(x) for x in outliers]
    }


# =========================
# Statistical depth (NEW)
# =========================
@app.post("/column/skewness/{col}")
def column_skewness(col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    require_numeric(df, col)
    series = df[col].dropna()
    skew = calculate_skewness(series)
    return {"column": col, "skewness": float(skew), "type": skew_type(skew)}


@app.post(
    "/column/kurtosis/{col}",
    summary="Excess kurtosis for a numeric column",
    tags=["Statistical Depth"]
)
def column_kurtosis(col: str, session_id: str = Query(...)):
    """
    Returns excess (Fisher) kurtosis and its interpretation:
    - leptokurtic (>1): heavy tails / sharp peak
    - platykurtic (<-1): thin tails / flat peak
    - mesokurtic: near-normal
    """
    df = get_df(session_id)
    require_numeric(df, col)
    result = calculate_kurtosis(df[col])
    return {"column": col, **result}


@app.post(
    "/stats/normality/{col}",
    summary="Normality test (Shapiro-Wilk or K-S)",
    tags=["Statistical Depth"]
)
def stats_normality(col: str, session_id: str = Query(...)):
    """
    Runs Shapiro-Wilk for n ≤ 5000, Kolmogorov-Smirnov for larger samples.
    Returns statistic, p-value, and a plain-English verdict.
    """
    df = get_df(session_id)
    require_numeric(df, col)
    return {"column": col, **normality_test(df[col])}


@app.post(
    "/column/entropy/{col}",
    summary="Shannon entropy for a categorical column",
    tags=["Statistical Depth"]
)
def column_entropy(col: str, session_id: str = Query(...)):
    """
    Returns entropy in bits, max possible entropy, normalized entropy [0-1],
    and an interpretation. Useful for understanding category diversity.
    """
    df = get_df(session_id)
    require_col(df, col)
    return {"column": col, **value_entropy(df[col])}


@app.post(
    "/correlation",
    summary="Correlation matrix — Pearson (numeric) and Cramér's V (categorical)",
    tags=["Statistical Depth"]
)
def correlation(session_id: str = Query(...)):
    """
    Returns two matrices:
    - `pearson`: Pearson r for all numeric column pairs
    - `cramers_v`: Cramér's V for all categorical column pairs
    """
    df = get_df(session_id)
    return correlation_matrix(df)


# =========================
# Outlier detection (NEW)
# =========================
@app.post(
    "/outliers/{col}",
    summary="Detect outliers using IQR or Z-score",
    tags=["Outlier Detection"]
)
def outliers(
    col: str,
    session_id: str = Query(...),
    method: str = Query("iqr", pattern="^(iqr|zscore)$")
):
    """
    Flags outliers and returns their row indices, values, and summary statistics.
    - `method=iqr` (default): flags values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
    - `method=zscore`: flags values where |z| > 3
    """
    df = get_df(session_id)
    require_numeric(df, col)
    return detect_outliers(df, col, method=method)


# =========================
# Imputation (NEW — apply it)
# =========================
@app.post(
    "/imputation/{col}/recommend",
    summary="Get imputation method recommendation",
    tags=["Imputation"]
)
def imputation_recommend(col: str, session_id: str = Query(...)):
    df = get_df(session_id)
    require_numeric(df, col)
    return recommend_imputation_method(df, col)


@app.post(
    "/imputation/{col}/apply",
    summary="Apply imputation to a column",
    tags=["Imputation"]
)
def imputation_apply(col: str, body: ImputeRequest, session_id: str = Query(...)):
    """
    Applies the chosen imputation method and returns the filled column values.
    Also updates the session's DataFrame in-place.

    Methods: `mean`, `median`, `mode`, `ffill`, `bfill`
    """
    df = get_df(session_id)
    require_col(df, col)
    try:
        result = apply_imputation(df, col, body.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update session DataFrame
    sessions[session_id][col] = result["values"]
    return result


# =========================
# Data quality (NEW)
# =========================
@app.post(
    "/suggest/dtypes",
    summary="Suggest better data types for each column",
    tags=["Data Quality"]
)
def suggest_data_types(session_id: str = Query(...)):
    """
    Heuristically inspects each column and recommends better dtypes:
    detects dates stored as strings, boolean-like columns, ID columns,
    numeric columns stored as objects, and high-cardinality candidates for `category`.
    """
    df = get_df(session_id)
    return suggest_dtypes(df)


# =========================
# ML readiness (NEW)
# =========================
@app.post(
    "/ml/encoding_suggestions",
    summary="Recommend encoding strategy for categorical columns",
    tags=["ML Readiness"]
)
def ml_encoding(session_id: str = Query(...)):
    """
    For each categorical column recommends:
    - One-hot encoding (low cardinality)
    - Ordinal / label encoding (binary or ordinal)
    - Target encoding or hashing (high cardinality)
    """
    df = get_df(session_id)
    return encoding_suggestions(df)


@app.post(
    "/ml/feature_importance/{target_col}",
    summary="Mutual information feature importance against a target column",
    tags=["ML Readiness"]
)
def ml_feature_importance(target_col: str, session_id: str = Query(...)):
    """
    Computes mutual information scores between all other columns and the target.
    Works for both classification (categorical target) and regression (numeric target).
    Returns a ranked list with high / medium / low importance labels.
    """
    df = get_df(session_id)
    require_col(df, target_col)
    result = feature_importance_proxy(df, target_col)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post(
    "/ml/class_balance/{col}",
    summary="Class distribution and imbalance check for a target column",
    tags=["ML Readiness"]
)
def ml_class_balance(col: str, session_id: str = Query(...)):
    """
    Shows class counts, ratios, and flags imbalance when any class < 20%.
    Includes a recommended remediation strategy when imbalance is detected.
    """
    df = get_df(session_id)
    require_col(df, col)
    return class_balance(df, col)


# =========================
# Auto EDA report (NEW)
# =========================
@app.post(
    "/report",
    summary="Full auto EDA report — one call, everything",
    tags=["Auto Report"]
)
async def report(session_id: str = Query(...)):
    """
    Runs all analyses in a single call and returns a unified JSON report including:
    - Overview (shape, memory, duplicates)
    - Per-column stats (nulls, skewness, kurtosis, normality, outliers, entropy)
    - Correlation matrices (Pearson + Cramér's V)
    - Dtype suggestions
    - Encoding suggestions
    """
    df = get_df(session_id)
    result = await asyncio.to_thread(full_eda_report, df)
    return result
