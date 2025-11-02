import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import setup_logging, read_csv_with_date, ensure_columns

logger = setup_logging("train_classifier")

REQUIRED_USER = ["code", "company", "name", "gender", "age"]
REQUIRED_FLIGHT = ["userCode", "from", "to", "flightType", "price", "time", "distance", "agency", "date"]
REQUIRED_HOTEL = ["userCode", "place", "days", "price", "total", "date"]

def engineer_user_features(user_df: pd.DataFrame, flights: pd.DataFrame, hotels: pd.DataFrame) -> pd.DataFrame:
    f_agg = flights.groupby("userCode").agg(
        flights_count=("userCode", "count"),
        avg_flight_price=("price", "mean"),
        avg_distance=("distance", "mean"),
        premium_share=("flightType", lambda s: (s == "premium").mean()),
        first_share=("flightType", lambda s: (s == "firstClass").mean()),
        economic_share=("flightType", lambda s: (s == "economic").mean()),
    ).reset_index().rename(columns={"userCode": "code"})

    top_to = flights.groupby(["userCode", "to"]).size().reset_index(name="cnt")
    top_to = top_to.sort_values(["userCode", "cnt"], ascending=[True, False]).drop_duplicates("userCode")
    top_to = top_to.rename(columns={"userCode": "code", "to": "preferred_dest"})
    
    h_agg = hotels.groupby("userCode").agg(
        hotels_count=("userCode", "count"),
        avg_hotel_price=("price", "mean"),
        avg_days=("days", "mean"),
    ).reset_index().rename(columns={"userCode": "code"})

    df = user_df.copy()
    df = df.merge(f_agg, on="code", how="left")
    df = df.merge(top_to[["code", "preferred_dest"]], on="code", how="left")
    df = df.merge(h_agg, on="code", how="left")

    df[["flights_count","avg_flight_price","avg_distance","premium_share","first_share","economic_share",
        "hotels_count","avg_hotel_price","avg_days"]] = \
        df[["flights_count","avg_flight_price","avg_distance","premium_share","first_share","economic_share",
            "hotels_count","avg_hotel_price","avg_days"]].fillna(0)

    return df

def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "travel-ml"))

    users = pd.read_csv(args.user_path)
    flights = read_csv_with_date(args.flight_path, date_cols=["date"])
    hotels = read_csv_with_date(args.hotel_path, date_cols=["date"])

    ensure_columns(users, REQUIRED_USER)
    ensure_columns(flights, REQUIRED_FLIGHT)
    ensure_columns(hotels, REQUIRED_HOTEL)

    df = engineer_user_features(users, flights, hotels)

    df = df[df["gender"].notna()]
    y = df["gender"].astype("category")

    features = [
        "company", "age", "flights_count", "avg_flight_price", "avg_distance",
        "premium_share", "first_share", "economic_share",
        "hotels_count", "avg_hotel_price", "avg_days",
        "preferred_dest"
    ]
    X = df[features]

    cat_cols = ["company", "preferred_dest"]
    num_cols = [c for c in features if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", clf)])

    with mlflow.start_run(run_name="train_gender_classifier"):
        n = len(df)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        logger.info(f"Accuracy: {acc:.3f} F1-macro: {f1:.3f}")
        logger.info("\n" + classification_report(y_test, preds))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(
            pipe, artifact_path="model",
            registered_model_name=args.register_name if args.register_name else None
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipe, "models/gender_classifier.pkl")
        mlflow.log_artifact("models/gender_classifier.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight_path", required=True)
    parser.add_argument("--hotel_path", required=True)
    parser.add_argument("--user_path", required=True)
    parser.add_argument("--register_name", type=str, default=None)
    args = parser.parse_args()
    main(args)