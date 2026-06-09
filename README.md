# Multi-Cast Time Series Forecasting Suite

This is not an officially supported Google product. This project is not eligible
for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

## Overview

The **Multi-Cast Forecasting Suite** is an enterprise-grade, multi-model time
series forecasting platform designed for scalable analytical modeling and
automated production batch execution.

Built around a domain-driven microservice architecture, the suite orchestrates
advanced predictive algorithms—including **Prophet**, Google's foundational
**TimesFM**, and BigQuery ML **ARIMA_PLUS**—into unified comparative
benchmarking dashboards. The automated pipeline is fully managed via Data
Version Control (DVC) and optimized for serverless, high-concurrency execution
on Google Cloud Run Jobs.

--------------------------------------------------------------------------------

## Project Architecture & Directory Layout

The repository adheres to clean software engineering conventions, strictly
separating root-level cloud infrastructure definitions from core domain modeling
packages.

```text
multi_cast/ (Project Root)
├── README.md                 # Global project documentation
├── requirements.txt          # Unified Python dependency specifications
├── Dockerfile                # Container definition for Cloud Build / standalone runtimes
├── cloud_run_job_main.py     # Serverless Cloud Run batch execution entrypoint
├── cloud_run_job_main_test.py# Unit test suite for the batch entrypoint
├── dvc.yaml                  # Main DVC pipeline orchestration DAG definition
├── params_base.yaml          # Immutable baseline pipeline hyperparameter template
├── env.yaml                  # Environment execution tracking configuration
├── cloud_run_deploy.env      # Dedicated Cloud Run Job resource allocation definitions
├── .gitignore
└── src/                      # Core Domain Packages
    ├── __init__.py
    │
    ├── common/               # Shared Infrastructure Layer
    │   ├── utils.py          # Common GCS, YAML, and runtime utilities
    │   └── utils_test.py
    │
    ├── pipelines/            # Data Engineering & Reporting Pipeline Layer
    │   ├── download_data.py  # Pure Python GCS raw data ingestion module
    │   ├── clean_data.py     # Pre-processing and outlier removal
    │   ├── split_train_test_data.py
    │   ├── upload_to_bq.py
    │   ├── join_forecasts.py # Multi-solution predictions consolidation
    │   └── plot_time_series.py # Macro time series reporting visualization
    │
    └── models/               # Autonomous Algorithmic Modeling & Self-Evaluation Layer
        ├── prophet/
        │   ├── forecast_prophet.py
        │   ├── generate_forecast_prophet.py
        │   └── evaluate_prophet_forecasts.py # Dedicated self-evaluation logic
        ├── timesfm/
        │   ├── forecast_timesfm.py
        │   ├── generate_forecast_timesfm.py
        │   └── evaluate_timesfm_forecasts.py
        └── arima_plus/
            ├── forecast_arima_plus.py
            ├── generate_forecast_arima_plus.py
            └── evaluate_arima_plus_forecasts.py
```

--------------------------------------------------------------------------------

## Prerequisites & Dependency Management

The platform requires **Python 3.12+**. Dependency management is optimized for
the lightning-fast `uv` package manager but remains fully compatible with
standard `pip`.

To install dependencies directly from the project root:

### Using uv (Recommended for 10x-50x installation acceleration)

```bash
uv pip install -r requirements.txt
```

### Or using standard pip

```bash
pip install -r requirements.txt
```

--------------------------------------------------------------------------------

## Configuration Hierarchy

The platform enforces a strict separation between immutable baseline defaults
and dynamic runtime overrides.

1.  **`params_base.yaml`**: The immutable, version-controlled baseline
    configuration declaring data storage paths, default execution tags, and
    algorithm hyperparameter boundaries.
2.  **`params.yaml` (Operational Override)**: An uncommitted, user-customized
    operational configuration file residing in the project root. Developers
    modify this file directly to alter execution tags, data paths, or specific
    hyperparameter boundaries for local runs.
3.  **`cloud_run_deploy.env`**: A dedicated Infrastructure-as-Code (IaC)
    configuration declaring production-grade Cloud Run Job resource limits (CPU,
    Memory, Parallelism, Retries, and Timeouts).

--------------------------------------------------------------------------------

## Local DVC Pipeline Execution

To execute the forecasting pipeline locally within your execution environment,
follow these steps:

### Step 1: Parameter Customization

To configure your operational pipeline, copy the immutable baseline
hyperparameter template `params_base.yaml` and save it as `params.yaml` in the
project root:

```bash
cp params_base.yaml params.yaml
```

Open `params.yaml` in your preferred editor and customize the execution tags,
data storage paths, or hyperparameter search boundaries to match your target
forecasting environment.

### Step 2: Native DVC Reproduction

Trigger the end-to-end forecasting DAG directly in your terminal:

```bash
dvc repro
```

*DVC will autonomously execute raw data ingestion (`download_data.py`), ETL
cleaning, parallel multi-model forecasting, and macro dashboard visualization.
All heavy execution caches and output artifacts are cleanly preserved.*

--------------------------------------------------------------------------------

## Cloud Run Serverless Batch Deployment

The project is engineered for seamless, high-concurrency serverless batch
execution on Google Cloud Run Jobs.

When deploying via the helper suite (`./export_and_deploy.sh`), a pristine
`sanitized_export/` directory is generated. The platform supports two primary
cloud deployment avenues:

### Option A (Recommended): Direct Cloud Run Jobs Source Deployment

Leveraging Google Cloud Serverless Buildpacks, the platform instantly recognizes
the injected `main.py` symlink and `.python-version` files, automatically
packaging and deploying the job without requiring a manual container build.

Resource allocations are dynamically loaded from `cloud_run_deploy.env`. You can
also override these allocations on the fly via environment variables:

```bash
# Deploy directly from the sanitized export context
JOB_MEMORY=8Gi gcloud run jobs deploy multi-cast-batch-job \
    --source sanitized_export/ \
    --region us-central1 \
    --cpu 2 \
    --memory 4Gi \
    --max-retries 1 \
    --parallelism 5 \
    --task-timeout 3600s
```

### Option B: Cloud Build Container Submission

To submit the build context to Google Cloud Build for custom container registry
archiving:

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REGISTRY/batch-job:latest sanitized_export/
```
