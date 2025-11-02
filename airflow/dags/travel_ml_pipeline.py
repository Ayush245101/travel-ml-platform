from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.train_regression import main as train_regression_main
from src.train_classifier import main as train_classifier_main
from src.train_recommender import main as train_recommender_main

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": [],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def train_regression_task():
    args = type("Args", (), {})()
    args.data_path = os.getenv("FLIGHT_CSV", "/usr/local/airflow/data/flight.csv")
    args.register_name = os.getenv("REG_MODEL_NAME", "FlightPriceModel")
    train_regression_main(args)

def train_classifier_task():
    args = type("Args", (), {})()
    args.flight_path = os.getenv("FLIGHT_CSV", "/usr/local/airflow/data/flight.csv")
    args.hotel_path = os.getenv("HOTEL_CSV", "/usr/local/airflow/data/hotal.csv")
    args.user_path = os.getenv("USER_CSV", "/usr/local/airflow/data/user.csv")
    args.register_name = os.getenv("CLS_MODEL_NAME", "GenderClassifier")
    train_classifier_main(args)

def train_recommender_task():
    args = type("Args", (), {})()
    args.flight_path = os.getenv("FLIGHT_CSV", "/usr/local/airflow/data/flight.csv")
    args.hotel_path = os.getenv("HOTEL_CSV", "/usr/local/airflow/data/hotal.csv")
    args.user_path = os.getenv("USER_CSV", "/usr/local/airflow/data/user.csv")
    args.output_path = os.getenv("RECOMMENDER_ARTIFACT", "/usr/local/airflow/models/recommender.pkl")
    train_recommender_main(args)

with DAG(
    dag_id="travel_ml_pipeline",
    default_args=default_args,
    description="Train and log models for travel ML project",
    schedule_interval="@weekly",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:

    train_regression = PythonOperator(
        task_id="train_regression",
        python_callable=train_regression_task
    )

    train_classifier = PythonOperator(
        task_id="train_gender_classifier",
        python_callable=train_classifier_task
    )

    train_recommender = PythonOperator(
        task_id="train_recommender",
        python_callable=train_recommender_task
    )

    train_regression >> train_classifier >> train_recommender