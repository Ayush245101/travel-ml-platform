import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import setup_logging, read_csv_with_date, ensure_columns, add_date_parts

logger = setup_logging("train_regression")

REQUIRED_COLS = ["from", "to", "flightType", "time", "distance", "agency", "date", "price"]

def build_pipeline(cat_cols, num_cols):
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe

def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "travel-ml"))

    df = read_csv_with_date(args.data_path, date_cols=["date"])
    ensure_columns(df, REQUIRED_COLS)

    df = add_date_parts(df, "date")

    target = "price"
    features = [
        "from", "to", "flightType", "time", "distance", "agency",
        "month", "dayofweek", "is_weekend"
    ]

    X = df[features]
    y = df[target]

    cat_cols = ["from", "to", "flightType", "agency"]
    num_cols = ["time", "distance", "month", "dayofweek", "is_weekend"]

    pipe = build_pipeline(cat_cols, num_cols)

    with mlflow.start_run(run_name="train_flight_regression"):
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_params({
            "n_estimators": 300,
            "random_state": 42
        })

        n = len(df)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        logger.info(f"RMSE: {rmse:.3f} R2: {r2:.3f}")

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name=args.register_name if args.register_name else None
        )

        os.makedirs("models", exist_ok=True)
        local_path = "models/regression_model.pkl"
        joblib.dump(pipe, local_path)
        mlflow.log_artifact(local_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to flight.csv")
    parser.add_argument("--register_name", type=str, default=None, help="MLflow registered model name")
    args = parser.parse_args()
    main(args)