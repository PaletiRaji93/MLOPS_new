"""
train.py
========
Fetches March 2025 features from Feast, trains 5 forecasting models to predict
DAM MCP, logs each model to MLflow, and saves the best model + metrics for DVC.

Models: ARIMA, Exponential Smoothing (Holt-Winters), Prophet, Gradient Boosting, XGBoost
Target: dam_mcp (Day Ahead Market Clearing Price)

Pipeline position:
  Feast (offline store) -> train.py -> models/model.pkl + metrics.json

Usage:
    python src/4_training/1_train.py
    python src/4_training/1_train.py --test-size 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from feast import FeatureStore  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

warnings.filterwarnings("ignore")

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# -- Config ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FEAST_REPO = PROJECT_ROOT / "my_feature_store" / "feature_repo"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "metrics.json"
TARGET_COL = "dam_mcp"
TIMESTAMP_COL = "event_timestamp"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = "dam_mcp_forecast"
REGISTERED_MODEL_NAME = "dam_mcp_forecast"

# Feast FeatureService that bundles all model features (defined once in
# feature_definitions.py). Every layer references this name instead of
# re-listing feature columns, so the feature set can never drift between
# training, evaluation, and serving.
FEATURE_SERVICE = "dam_mcp_forecast_v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s")
log = logging.getLogger("train")
logging.getLogger("feast.infra.registry.base_registry").setLevel(logging.WARNING)
logging.getLogger("feast").setLevel(logging.WARNING)


# -- CLI ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 5 forecasting models")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data to use as test set (default: 0.2)")
    return parser.parse_args()


# -- Step 1: Fetch features from Feast --------------------------------------
def fetch_training_data() -> pd.DataFrame:
    store = FeatureStore(repo_path=str(FEAST_REPO))
    source_path = FEAST_REPO / "data" / "march_2025_features.parquet"
    source_df = pd.read_parquet(source_path)
    entity_df = source_df[["block_id", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    feature_service = store.get_feature_service(FEATURE_SERVICE)
    log.info(f"Fetching feature service '{FEATURE_SERVICE}' for {len(entity_df)} blocks from Feast...")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    ).to_df()

    log.info(f"Fetched: {training_df.shape[0]} rows, {training_df.shape[1]} columns")
    return training_df


# -- Step 2: Prepare data ---------------------------------------------------
def prepare_data(df: pd.DataFrame, test_size: float):
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    timestamps = df[[TIMESTAMP_COL]]
    y = df[TARGET_COL]

    exclude_cols = {TARGET_COL, TIMESTAMP_COL, "block_id"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    for col in num_cols:
        if pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype(float)

    # Time-based split
    n = max(1, int(len(X) * (1 - test_size)))
    X_train, X_test = X.iloc[:n], X.iloc[n:]
    y_train, y_test = y.iloc[:n], y.iloc[n:]
    ts_train, ts_test = timestamps.iloc[:n], timestamps.iloc[n:]

    log.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    log.info(f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

    return X_train, X_test, y_train, y_test, ts_train, ts_test, cat_cols, num_cols


# -- Step 3: Helpers ---------------------------------------------------------
def build_preprocessor(cat_cols: List[str], num_cols: List[str]):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def log_plots(y_true, y_pred, run_name: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y_true, label="actual", linewidth=1)
        ax.plot(y_pred, label="predicted", linewidth=1)
        ax.set_title(f"{run_name}: Predictions vs Actuals")
        ax.legend()
        fig.tight_layout()
        p = Path(tmpdir) / "pred_vs_actual.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(p), artifact_path="plots")

        fig, ax = plt.subplots(figsize=(8, 3))
        residuals = np.array(y_true) - np.array(y_pred)
        ax.plot(residuals, linewidth=1)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title(f"{run_name}: Residuals")
        fig.tight_layout()
        p = Path(tmpdir) / "residuals.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(p), artifact_path="plots")


# -- Step 4: Train 5 models -------------------------------------------------
def train_arima(y_train, y_test) -> Dict[str, float]:
    log.info("[1/5] Training ARIMA ...")
    with mlflow.start_run(run_name="arima"):
        order = (5, 1, 2)
        mlflow.log_params({"order_p": order[0], "order_d": order[1], "order_q": order[2]})

        model = ARIMA(y_train.values, order=order)
        fitted = model.fit()

        preds = fitted.forecast(steps=len(y_test))
        metrics = compute_metrics(y_test.values, preds)
        mlflow.log_metrics(metrics)
        log_plots(y_test.values, preds, "arima")
        mlflow.log_text(fitted.summary().as_text(), "arima_summary.txt")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "arima_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(fitted, f)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        log.info(f"  ARIMA -> RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  MAPE={metrics['mape']:.2f}%")
    return metrics


def train_exp_smoothing(y_train, y_test) -> Dict[str, float]:
    log.info("[2/5] Training Exponential Smoothing (Holt-Winters) ...")
    with mlflow.start_run(run_name="exponential_smoothing"):
        params = {"trend": "add", "seasonal": "add", "seasonal_periods": 96}
        mlflow.log_params(params)

        model = ExponentialSmoothing(
            y_train.values,
            trend=params["trend"],
            seasonal=params["seasonal"],
            seasonal_periods=params["seasonal_periods"],
        )
        fitted = model.fit(optimized=True)

        preds = fitted.forecast(steps=len(y_test))
        metrics = compute_metrics(y_test.values, preds)
        mlflow.log_metrics(metrics)
        log_plots(y_test.values, preds, "exponential_smoothing")
        mlflow.log_text(str(fitted.summary()), "ets_summary.txt")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "exp_smoothing_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(fitted, f)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        log.info(f"  ETS   -> RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  MAPE={metrics['mape']:.2f}%")
    return metrics


def train_prophet(ts_train, y_train, ts_test, y_test) -> Dict[str, float]:
    log.info("[3/5] Training Prophet ...")
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("  Prophet not available (needs CmdStan). Skipping.")
        return None

    with mlflow.start_run(run_name="prophet"):
        df_train = pd.DataFrame({"ds": ts_train.iloc[:, 0].values, "y": y_train.values})
        df_test = pd.DataFrame({"ds": ts_test.iloc[:, 0].values, "y": y_test.values})

        params = {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "seasonality_mode": "multiplicative",
        }
        mlflow.log_params(params)

        m = Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
        )
        m.fit(df_train)

        forecast = m.predict(df_test[["ds"]])
        preds = forecast["yhat"].values
        metrics = compute_metrics(y_test.values, preds)
        mlflow.log_metrics(metrics)
        log_plots(y_test.values, preds, "prophet")

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = m.plot_components(forecast)
            comp_path = Path(tmpdir) / "prophet_components.png"
            fig.savefig(comp_path, dpi=100)
            plt.close(fig)
            mlflow.log_artifact(str(comp_path), artifact_path="plots")

            model_path = Path(tmpdir) / "prophet_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(m, f)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        log.info(f"  Prophet -> RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  MAPE={metrics['mape']:.2f}%")
    return metrics


def train_gradient_boosting(X_train, y_train, X_test, y_test, cat_cols, num_cols) -> Tuple[Dict[str, float], Pipeline]:
    log.info("[4/5] Training Gradient Boosting ...")
    with mlflow.start_run(run_name="gradient_boosting"):
        preprocessor = build_preprocessor(cat_cols, num_cols)
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=42,
            )),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        metrics = compute_metrics(y_test.values, preds)
        mlflow.log_metrics(metrics)
        mlflow.log_params({"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8})
        log_plots(y_test.values, preds, "gradient_boosting")

        sig = infer_signature(X_train, pipe.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipe, artifact_path="model", signature=sig,
            input_example=X_train.head(5),
        )
        run_id = mlflow.active_run().info.run_id
        log.info(f"  GBR   -> RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  MAPE={metrics['mape']:.2f}%")
    return metrics, pipe, run_id


def train_xgboost(X_train, y_train, X_test, y_test, cat_cols, num_cols) -> Tuple[Dict[str, float], Pipeline]:
    log.info("[5/5] Training XGBoost ...")
    with mlflow.start_run(run_name="xgboost"):
        preprocessor = build_preprocessor(cat_cols, num_cols)
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=1,
            )),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        metrics = compute_metrics(y_test.values, preds)
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.8, "colsample_bytree": 0.8,
        })
        log_plots(y_test.values, preds, "xgboost")

        sig = infer_signature(X_train, pipe.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipe, artifact_path="model", signature=sig,
            input_example=X_train.head(5),
        )
        run_id = mlflow.active_run().info.run_id
        log.info(f"  XGB   -> RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  MAPE={metrics['mape']:.2f}%")
    return metrics, pipe, run_id


# -- Step 5: Save best model + metrics for DVC ------------------------------
def save_best(all_metrics: Dict[str, Dict], best_pipe):
    best_name = min(all_metrics, key=lambda k: all_metrics[k]["rmse"])
    best_metrics = all_metrics[best_name]

    log.info(f"Best model: {best_name} (RMSE={best_metrics['rmse']:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipe, f)
    log.info(f"Model saved: {MODEL_PATH}")

    output_metrics = {
        "best_model": best_name,
        "val_rmse": best_metrics["rmse"],
        "val_mae": best_metrics["mae"],
        "val_mape": best_metrics["mape"],
        "all_models": all_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output_metrics, f, indent=2)
    log.info(f"Metrics saved: {METRICS_PATH}")

    return best_name, best_metrics


# -- Main --------------------------------------------------------------------
def main():
    from src.lineage import lineage_run, ds

    args = parse_args()

    log.info("=" * 55)
    log.info("TRAINING - DAM MCP Prediction (5 models)")
    log.info("=" * 55)

    with lineage_run("train",
        inputs=[ds("feast/march_2025_features.parquet")],
        outputs=[ds("models/model.pkl"), ds("metrics.json")],
    ):
        # 1. Fetch from Feast
        df = fetch_training_data()

        # 2. Prepare & split
        X_train, X_test, y_train, y_test, ts_train, ts_test, cat_cols, num_cols = prepare_data(
            df, args.test_size
        )

        # 3. MLflow setup
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        log.info(f"MLflow tracking: {MLFLOW_TRACKING_URI} | experiment: {MLFLOW_EXPERIMENT}")

        # 4. Train all 5 models
        all_metrics = {}
        pipelines = {}

        all_metrics["arima"] = train_arima(y_train, y_test)

        all_metrics["exponential_smoothing"] = train_exp_smoothing(y_train, y_test)

        prophet_metrics = train_prophet(ts_train, y_train, ts_test, y_test)
        if prophet_metrics is not None:
            all_metrics["prophet"] = prophet_metrics

        run_ids = {}

        all_metrics["gradient_boosting"], pipelines["gradient_boosting"], run_ids["gradient_boosting"] = \
            train_gradient_boosting(X_train, y_train, X_test, y_test, cat_cols, num_cols)

        all_metrics["xgboost"], pipelines["xgboost"], run_ids["xgboost"] = \
            train_xgboost(X_train, y_train, X_test, y_test, cat_cols, num_cols)

        # 5. Save best model (only sklearn pipelines can be saved as pkl)
        best_name = min(all_metrics, key=lambda k: all_metrics[k]["rmse"])
        if best_name in pipelines:
            best_pipe = pipelines[best_name]
        else:
            # If a univariate model wins, fall back to best sklearn pipeline
            sklearn_best = min(
                (k for k in all_metrics if k in pipelines),
                key=lambda k: all_metrics[k]["rmse"],
            )
            log.info(f"Best overall is {best_name} (univariate), saving best sklearn pipeline: {sklearn_best}")
            best_name = sklearn_best
            best_pipe = pipelines[sklearn_best]

        save_best(all_metrics, best_pipe)

        # 6. Register ONLY the best model in the registry (1 version per training run)
        best_run_id = run_ids[best_name]
        model_uri = f"runs:/{best_run_id}/model"
        result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        log.info(f"Registered best model: {best_name} as {REGISTERED_MODEL_NAME} v{result.version}")

    # Summary
    log.info("")
    log.info("=" * 55)
    log.info("RESULTS SUMMARY")
    log.info("-" * 55)
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1]["rmse"]):
        log.info(f"  {name:25s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%")
    log.info("=" * 55)
    log.info(f"All 5 models logged to MLflow at {MLFLOW_TRACKING_URI}")
    log.info(f"Experiment: '{MLFLOW_EXPERIMENT}' | Registry: '{REGISTERED_MODEL_NAME}'")


if __name__ == "__main__":
    main()
