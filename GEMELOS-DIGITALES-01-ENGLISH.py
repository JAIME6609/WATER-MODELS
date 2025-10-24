# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 18:26:21 2025

@author: Asus.S510UNR
"""

# -*- coding: utf-8 -*-
"""
mDTO — Minimal Viable Ocean Digital Twin (operational conclusion)
Architecture: Ingest -> Preparation -> Modeling -> Evaluation -> Visualization/Deployment
Outputs: metrics, tables, interactive charts, and Excel files

Key components:
- Famous dataset (attempt to load from a public repository); synthetic fallback if offline.
- Feature engineering (lags, seasonal, calendar-like encodings).
- Baseline models: Persistence and Monthly Climatology.
- ML model: Gradient Boosting Regressor for forecasting (with quantile variants for PIs).
- Metrics: MAE, MSE, RMSE, MAPE, R2, Skill vs climatology (1 - RMSE_model/RMSE_clim).
- Reports: DataFrames to Excel (predictions, metrics, feature importance).
- Dashboard: Plotly + Dash with time series, residuals, error distribution, and uncertainty bands.

Environment (typical):
    pip install pandas numpy scikit-learn statsmodels plotly dash openpyxl

Dataset note:
    - Tries to load a well-known public monthly “global temperature” (land+ocean) dataset
      maintained by the open-data community (raw CSV on GitHub).
    - If no internet is available, it generates a synthetic series with annual seasonality + trend + noise
      so the entire pipeline remains reproducible in offline mode.
"""

# --- Imports
import os
import io
import sys
import math
import json
import warnings
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd

# Modeling & metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Visualization
import plotly.graph_objects as go
import plotly.express as px

# Dashboard
from dash import Dash, html, dcc, dash_table, Input, Output

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# 1) General utilities
# -----------------------------
def try_load_famous_global_temp() -> pd.DataFrame:
    """
    Attempts to load a FAMOUS and OPEN monthly global temperature dataset
    (land+ocean, “global-temp monthly” maintained by the data packages community).
    If this fails (e.g., offline), returns None.

    Expected CSV structure:
        Date, Source, Mean
    with Date formatted as YYYY-MM-01 and Mean the anomaly (°C).

    Reference URL (subject to public availability):
        https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv
    """
    import urllib.request
    url = "https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read()
        df = pd.read_csv(io.BytesIO(data))
        # Filter "GCAG" (Global Combined Land+Ocean) if available
        if "Source" in df.columns and "Mean" in df.columns:
            df = df[df["Source"].astype(str).str.upper().eq("GCAG")].copy()
        # Normalize columns
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
        else:
            # If the CSV changed the date column name
            first_date_col = [c for c in df.columns if "date" in c.lower()]
            df["date"] = pd.to_datetime(df[first_date_col[0]])
        df = df.sort_values("date").reset_index(drop=True)
        # Rename measurement column to 'value' (anomaly °C)
        if "Mean" in df.columns:
            df.rename(columns={"Mean": "value"}, inplace=True)
        else:
            # Fallback: first numerical column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df.rename(columns={num_cols[0]: "value"}, inplace=True)
        df = df[["date", "value"]].dropna()
        # Ensure monthly frequency (Month Start)
        df = df.set_index("date").asfreq("MS").reset_index()
        return df
    except Exception:
        return None


def generate_synthetic_sst(n_years: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic monthly time series akin to a 'global SST anomaly':
    trend + annual seasonality + noise. Units are arbitrary (°C).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1990-01-01", periods=12 * n_years, freq="MS")
    t = np.arange(len(dates))
    trend = 0.002 * t  # slight positive trend
    seasonal = 0.15 * np.sin(2 * np.pi * t / 12.0)  # annual pattern
    noise = rng.normal(0, 0.05, size=len(dates))
    values = trend + seasonal + noise
    df = pd.DataFrame({"date": dates, "value": values})
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds temporal and climatology-like features: month, year, seasonal sin/cos, lags, and rolling stats."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # Seasonal encoding
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    # Lags
    for lag in [1, 3, 6, 12]:
        df[f"lag_{lag}"] = df["value"].shift(lag)
    # Rolling stats
    for w in [3, 6, 12]:
        df[f"rollmean_{w}"] = df["value"].rolling(w).mean()
        df[f"rollstd_{w}"] = df["value"].rolling(w).std()
    # Monthly climatology (historical average by month up to t-1)
    df["clim_month"] = (
        df.groupby("month")["value"]
          .apply(lambda s: s.shift(12).expanding().mean())
          .reset_index(level=0, drop=True)
    )
    return df


def train_test_split_time(df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-order-respecting train/test split."""
    n = len(df)
    n_test = int(math.ceil(n * test_frac))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_clim: np.ndarray = None) -> Dict[str, float]:
    """Computes metrics and a skill score relative to climatology (if provided)."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
    r2 = r2_score(y_true, y_pred)
    out = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE_%": mape, "R2": r2}
    if y_clim is not None:
        rmse_clim = math.sqrt(mean_squared_error(y_true, y_clim))
        skill = 1.0 - (rmse / (rmse_clim + 1e-12))
        out["RMSE_climatology"] = rmse_clim
        out["Skill_vs_Clim"] = skill
    return out


def build_quantile_models(random_state: int = 42):
    """
    Gradient Boosting models in quantile mode for predictive intervals (10% and 90%)
    and a 'squared_error' model for the mean forecast.
    """
    model_p50 = GradientBoostingRegressor(random_state=random_state, loss="squared_error")
    model_p10 = GradientBoostingRegressor(random_state=random_state, loss="quantile", alpha=0.10)
    model_p90 = GradientBoostingRegressor(random_state=random_state, loss="quantile", alpha=0.90)
    return model_p10, model_p50, model_p90


# -----------------------------
# 2) Data ingest
# -----------------------------
df = try_load_famous_global_temp()
source_note = "Famous public dataset: Global Land-Ocean Monthly Temperature (GCAG)"
if df is None:
    df = generate_synthetic_sst(n_years=40, seed=123)
    source_note = "Synthetic fallback dataset (offline mode)"

df = df.sort_values("date").reset_index(drop=True)

# -----------------------------
# 3) Preparation & feature engineering
# -----------------------------
df_feat = add_time_features(df)
df_feat = df_feat.dropna().reset_index(drop=True)

# Target and features
target = "value"
feature_cols = [c for c in df_feat.columns if c not in ["date", "value"]]

# Time split
train_df, test_df = train_test_split_time(df_feat, test_frac=0.2)

X_train, y_train = train_df[feature_cols], train_df[target].values
X_test,  y_test  = test_df[feature_cols],  test_df[target].values

# -----------------------------
# 4) Baselines
# -----------------------------
# Persistence: y_hat_t = y_{t-1}
yhat_persist = test_df["lag_1"].values

# Monthly climatology (already computed in 'clim_month'); uses history-defined values
yhat_clim = test_df["clim_month"].values

# Baseline metrics
metrics_persist = compute_metrics(y_test, yhat_persist, y_clim=yhat_clim)
metrics_clim   = compute_metrics(y_test, yhat_clim, y_clim=yhat_clim)  # skill vs itself will be 0

# -----------------------------
# 5) ML model (point forecast & intervals)
# -----------------------------
model_p10, model_p50, model_p90 = build_quantile_models(random_state=42)

model_p50.fit(X_train, y_train)
model_p10.fit(X_train, y_train)
model_p90.fit(X_train, y_train)

yhat_ml   = model_p50.predict(X_test)
yhat_q10  = model_p10.predict(X_test)
yhat_q90  = model_p90.predict(X_test)

# ML metrics
metrics_ml = compute_metrics(y_test, yhat_ml, y_clim=yhat_clim)

# Feature importance (impurity-based)
imp = pd.Series(model_p50.feature_importances_, index=feature_cols).sort_values(ascending=False)
feat_importance_df = imp.reset_index()
feat_importance_df.columns = ["feature", "importance"]

# -----------------------------
# 6) Tables & reports (Excel)
# -----------------------------
results_df = test_df[["date", "value"]].copy()
results_df["y_persistence"] = yhat_persist
results_df["y_climatology"] = yhat_clim
results_df["y_ml"] = yhat_ml
results_df["y_q10"] = yhat_q10
results_df["y_q90"] = yhat_q90
results_df["residual_ml"] = results_df["value"] - results_df["y_ml"]

metrics_table = pd.DataFrame([
    {"Model": "Persistence", **metrics_persist},
    {"Model": "Climatology", **metrics_clim},
    {"Model": "ML (GBR, mean)", **metrics_ml},
])

save_dir = "./mDTO_outputs"
os.makedirs(save_dir, exist_ok=True)

# Save to Excel
pred_xlsx   = os.path.join(save_dir, "predictions_and_intervals.xlsx")
metrics_xlsx = os.path.join(save_dir, "metrics.xlsx")
featimp_xlsx = os.path.join(save_dir, "feature_importance.xlsx")

with pd.ExcelWriter(pred_xlsx, engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="predictions")

with pd.ExcelWriter(metrics_xlsx, engine="openpyxl") as writer:
    metrics_table.to_excel(writer, index=False, sheet_name="metrics")

with pd.ExcelWriter(featimp_xlsx, engine="openpyxl") as writer:
    feat_importance_df.to_excel(writer, index=False, sheet_name="importance")

# -----------------------------
# 7) Plotly figures
# -----------------------------
def fig_series():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df["date"], y=results_df["value"],
                             mode="lines", name="Observed"))
    fig.add_trace(go.Scatter(x=results_df["date"], y=results_df["y_ml"],
                             mode="lines", name="Forecast (ML)"))
    fig.add_trace(go.Scatter(x=results_df["date"], y=results_df["y_persistence"],
                             mode="lines", name="Persistence", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=results_df["date"], y=results_df["y_climatology"],
                             mode="lines", name="Climatology", line=dict(dash="dot")))
    # Uncertainty band (q10–q90)
    fig.add_trace(go.Scatter(
        x=pd.concat([results_df["date"], results_df["date"][::-1]]),
        y=pd.concat([results_df["y_q90"], results_df["y_q10"][::-1]]),
        fill="toself",
        fillcolor="rgba(0, 0, 200, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="PI 10–90"
    ))
    fig.update_layout(
        title="Time Series: Observed vs Models (with Uncertainty Band)",
        xaxis_title="Date",
        yaxis_title="Value / Anomaly (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def fig_residuals():
    fig = px.histogram(results_df, x="residual_ml", nbins=30, opacity=0.8)
    fig.update_layout(
        title="Residual Distribution (ML Model)",
        xaxis_title="Residual",
        yaxis_title="Frequency"
    )
    return fig

def fig_feature_importance():
    fig = px.bar(feat_importance_df.head(15), x="feature", y="importance")
    fig.update_layout(
        title="Feature Importance (Top 15)",
        xaxis_title="Feature",
        yaxis_title="Importance",
        xaxis_tickangle=-35
    )
    return fig

def fig_pred_vs_obs():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df["value"], y=results_df["y_ml"],
        mode="markers", name="Obs vs ML"
    ))
    # Identity line
    lims = [float(results_df[["value", "y_ml"]].min().min()),
            float(results_df[["value", "y_ml"]].max().max())]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Identity", line=dict(dash="dash")))
    fig.update_layout(
        title="ML Forecast vs Observed",
        xaxis_title="Observed",
        yaxis_title="Predicted (ML)",
    )
    return fig

fig1 = fig_series()
fig2 = fig_residuals()
fig3 = fig_feature_importance()
fig4 = fig_pred_vs_obs()

# Export figures to standalone HTML (useful for reports)
fig1.write_html(os.path.join(save_dir, "fig_series.html"))
fig2.write_html(os.path.join(save_dir, "fig_residuals.html"))
fig3.write_html(os.path.join(save_dir, "fig_importance.html"))
fig4.write_html(os.path.join(save_dir, "fig_pred_vs_obs.html"))

# -----------------------------
# 8) Dash dashboard
# -----------------------------
app = Dash(__name__, title="mDTO — Ocean Digital Twin (Conclusion)")
app.layout = html.Div([
    html.H2("mDTO — Ocean Digital Twin: Short-Term Forecasting & Evaluation"),
    html.P(f"Data source: {source_note}"),
    html.Hr(),

    html.H4("Metrics Table"),
    dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in metrics_table.columns],
        data=metrics_table.round(6).to_dict("records"),
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold"},
        page_size=10
    ),

    html.H4("Time Series and Uncertainty Bands"),
    dcc.Graph(figure=fig1),

    html.Div([
        html.Div([html.H4("Residual Distribution (ML)"),
                  dcc.Graph(figure=fig2)], style={"width": "48%", "display": "inline-block"}),
        html.Div([html.H4("Feature Importance"),
                  dcc.Graph(figure=fig3)], style={"width": "48%", "display": "inline-block", "float": "right"}),
    ]),

    html.H4("Scatter: Observed vs Predicted (ML)"),
    dcc.Graph(figure=fig4),

    html.Hr(),
    html.H4("Predictions View (last 60 records)"),
    dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in results_df.columns],
        data=results_df.tail(60).round(6).to_dict("records"),
        style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "scroll"},
        page_size=15
    ),

    html.Hr(),
    html.P("Generated files (folder ./mDTO_outputs):"),
    html.Ul([
        html.Li("predictions_and_intervals.xlsx"),
        html.Li("metrics.xlsx"),
        html.Li("feature_importance.xlsx"),
        html.Li("fig_series.html"),
        html.Li("fig_residuals.html"),
        html.Li("fig_importance.html"),
        html.Li("fig_pred_vs_obs.html"),
    ]),
])

if __name__ == "__main__":
    # To run the dashboard locally:
    # python this_file.py
    # Then open http://127.0.0.1:2324 in your browser.
    print("The application is running at: http://127.0.0.1:2324")
    app.run(debug=False, port=2324)
