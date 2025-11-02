# Travel ML Platform

An end-to-end machine learning project that analyzes travel behavior, trains multiple models, and exposes predictions and insights through both REST and Streamlit interfaces. The repository bundles model training code, scheduled pipelines, web APIs, UI dashboards, and deployment automation via Docker, Jenkins, and Kubernetes.

---

## 🌍 What this project delivers

- **Predictive modeling** – RandomForest regression to estimate flight prices, a gender classifier, and a content-based hotel recommender.
- **Automated training** – An Apache Airflow DAG (`airflow/dags/travel_ml_pipeline.py`) orchestrates all training jobs on a weekly cadence.
- **Model tracking** – MLflow experiments/logs stored under `mlruns/`, with artifacts persisted in `models/` for local reuse.
- **Serving options** – A Flask API (`api/app.py`) for programmatic predictions and a Streamlit dashboard (`streamlit_app/app.py`) for interactive insights.
- **MLOps ready** – Docker image, Jenkins pipeline, and Kubernetes manifests for production deployments, plus Git-based CI hooks.


## 📁 Repository layout

```
.
├── airflow/              # Airflow DAG wiring model training jobs
├── api/                  # Flask REST API for flight price prediction
├── data/                 # Sample CSV datasets (flight, hotel, user)
├── k8s/                  # Kubernetes deployment, service, and HPA specs
├── models/               # Serialized artifacts (regression, classifier, recommender)
├── src/                  # Training scripts and shared utilities
├── streamlit_app/        # Streamlit UI for recommendations & analytics
├── tests/                # Pytest suite for regression pipeline sanity checks
├── Dockerfile            # Container build definition (Gunicorn + API)
├── Jenkinsfile           # CI/CD pipeline (lint, test, image build, deploy)
└── requirements.txt      # Python dependencies
```


## 📊 Datasets

| File | Description |
|------|-------------|
| `data/flight.csv` | Historical flight transactions with price, route, and metadata. |
| `data/hotal.csv`  | Hotel stays per user (name, place, duration, spend). |
| `data/user.csv`   | Traveler demographics and identifiers linking to flight/hotel records. |

The training scripts expect these CSVs. Update environment variables (`FLIGHT_CSV`, `HOTEL_CSV`, `USER_CSV`) to point at alternate locations when needed.


## ⚙️ Local setup

```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional extras:
- Copy any sensitive configuration into a `.env` file (for the API or Streamlit). Default paths are baked in, so this is only necessary for overrides.
- Launch MLflow UI with `mlflow ui --backend-store-uri mlruns` to inspect historical runs.


## 🧠 Training the models

Each script can run standalone; MLflow will capture parameters, metrics, and artifacts automatically.

```bash
# Flight price regression model
python -m src.train_regression --data_path data/flight.csv

# Gender classifier
python -m src.train_classifier \
  --flight_path data/flight.csv \
  --hotel_path data/hotal.csv \
  --user_path data/user.csv

# Hotel recommender artifact
python -m src.train_recommender \
  --flight_path data/flight.csv \
  --hotel_path data/hotal.csv \
  --user_path data/user.csv \
  --output_path models/recommender.pkl
```

*Artifacts* land in `models/` by default and are re-used by the API and Streamlit app.


## 🔁 Automated pipeline (Airflow)

`airflow/dags/travel_ml_pipeline.py` orchestrates the three training tasks sequentially:

1. `train_regression` – Fits the flight price model and logs to MLflow.
2. `train_gender_classifier` – Generates demographic classifier metrics/artifacts.
3. `train_recommender` – Builds the hotel recommendation artifact.

Deploy the DAG to an Airflow instance by copying the file into your Airflow DAGs directory. The DAG reads file paths and model names from environment variables to stay portable.


## 🧾 Model tracking (MLflow)

- Tracking URI defaults to `file:./mlruns`.
- Experiment name defaults to `travel-ml`.
- Artifacts (pickled pipelines, metrics, params) are visible through the MLflow UI or CLI.
- Configure `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` to point at a remote tracking server if desired.


## 🚀 Serving options

### Flask REST API

```bash
python -m src.train_regression --data_path data/flight.csv   # ensure model exists
python api/app.py
```

- Base URL: `http://localhost:8080`
- Health check: `GET /health`
- Prediction: `POST /predict` with JSON payload containing `from`, `to`, `flightType`, `time`, `distance`, `agency`, `date`.
- Environment variables: `MODEL_LOCAL_PATH`, `MLFLOW_MODEL_URI`, `API_HOST`, `API_PORT`.
- Production mode (via Docker) uses Gunicorn with two workers by default.

### Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py --server.port 8501
```

Features include user-level insights, spend metrics, and top-N hotel recommendations leveraging the recommender artifact. Configure artifact/data paths via environment variables (`RECOMMENDER_ARTIFACT`, `FLIGHT_CSV`, `HOTEL_CSV`, `USER_CSV`).


## ✅ Testing & linting

```bash
pytest
flake8 src api streamlit_app
```

The bundled tests (`tests/test_regression_features.py`) validate pipeline construction and basic predictions; extend as you expand model coverage.


## 🐳 Containerization

The `Dockerfile` builds a slim Python 3.11 image, installs dependencies, copies the API & models, and serves via Gunicorn.

```bash
docker build -t travel-ml-api:latest .
docker run -p 8080:8080 travel-ml-api:latest
```

Mount custom models or data by volume-mounting to `/app/models` or injecting environment variables during `docker run`.


## ⚙️ CI/CD with Jenkins

`Jenkinsfile` stages:

1. **Checkout** repository.
2. **Setup Python** virtualenv and install deps.
3. **Lint** with `flake8`.
4. **Test** with `pytest`.
5. **Build Docker image** tagged with build number.
6. **Push image** to Docker Hub using stored credentials.
7. **Deploy** updated manifests to Kubernetes (updates image tag inline, applies deployment/service/HPA).

Artifacts from the `models/` directory are archived after each run.


## ☸️ Kubernetes deployment

- `k8s/deployment.yaml` – Runs two replicas of the API container with readiness/liveness probes hitting `/health`. Environment variables point either to a bundled model or an MLflow registry (`models:/FlightPriceModel/Production`).
- `k8s/service.yaml` – Exposes the deployment as a ClusterIP on port 80 ➜ 8080.
- `k8s/hpa.yaml` – Horizontal Pod Autoscaler scales between 2–10 pods, targeting 70% CPU utilization.

Apply manifests after updating the container image reference:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```


## 🔭 Observability & monitoring

- **Logs** – Standard output from the API and Streamlit apps provide runtime insight. Consider plugging into centralized logging (e.g., Stackdriver, ELK) when deployed.
- **Metrics** – Track ML metrics through MLflow; for infra-level metrics add Prometheus scraping or cloud-native monitoring to the Kubernetes deployment.


## 🧭 Development tips & roadmap

- Remove deprecated `infer_datetime_format` arguments in `src/utils.py` to silence pandas warnings.
- Consider shuffling/splitting data before training to avoid overly optimistic metrics (current split takes the first 80%).
- Expand the pytest suite with checks for classifier/recommender output quality and API contract tests.
- Add CI gates for formatting (e.g., `black`) and type checking (`mypy` or `pyright`).
- Streamlit supports secrets management; move sensitive configs out of code by leveraging `.streamlit/secrets.toml`.


## 🙌 Contributing

1. Fork & clone the repository.
2. Create a feature branch: `git checkout -b feature/my-change`.
3. Run lint/tests before committing: `flake8` & `pytest`.
4. Submit a pull request with a summary of changes and validation evidence.

---

**Happy traveling & modeling!** Whether you are iterating locally, orchestrating in Airflow, or deploying to Kubernetes, this project provides a full-stack template for building and operating travel-focused ML systems.