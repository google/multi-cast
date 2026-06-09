# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates predictive segments leveraging TimesFM models via BigQuery queries.

Abstracts target dataset references avoiding multiple hardcoded segment arrays.
"""

from datetime import timedelta
import os

from absl import logging
from google.cloud import bigquery
import pandas as pd

from common import utils

SOLUTION = "timesfm"


def build_timesfm_query(
    project_id: str,
    dataset_id: str,
    table_id: str,
    cutoff_date: str,
    horizon: int,
    confidence_level: float,
    date_col: str,
    val_col: str,
) -> str:
  """Constructs the BigQuery BQML TimesFM forecasting query string.

  Args:
      project_id: Target GCP project identifier.
      dataset_id: Target BigQuery dataset identifier.
      table_id: Source table name.
      cutoff_date: Upper bound date filtering training data.
      horizon: Number of future steps to predict.
      confidence_level: Prediction interval confidence level setting.
      date_col: Configured date column name.
      val_col: Configured prediction value column name.

  Returns:
      Formatted BigQuery SQL query string.
  """
  return f"""
        SELECT
            * EXCEPT(forecast_value, prediction_interval_lower_bound, prediction_interval_upper_bound),
            GREATEST(forecast_value, 0.0001) AS forecast_value,
            GREATEST(prediction_interval_lower_bound, 0.0001) AS prediction_interval_lower_bound,
            GREATEST(prediction_interval_upper_bound, 0.0001) AS prediction_interval_upper_bound
        FROM AI.FORECAST(
            (
                SELECT {date_col}, {val_col}
                FROM `{project_id}.{dataset_id}.{table_id}`
                WHERE {date_col} < '{cutoff_date}'
            ),
            horizon => {horizon},
            confidence_level => {confidence_level},
            timestamp_col => '{date_col}',
            data_col => '{val_col}'
        );
    """


def execute_timesfm_forecast(config_path: str = "params.yaml") -> None:
  """Main execution method preparing target parameters and executing remote BigQuery predictions.

  Args:
      config_path: Path to the YAML configuration file.

  Raises:
      ValueError: If mandatory BigQuery configuration parameters are missing.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)
  date_col = utils.get_date_col(params)
  val_col = utils.get_val_col(params)

  # Unpack model boundaries and connection references securely
  tf_config = params.get("forecast_timesfm", {})
  prediction_window = int(tf_config.get("prediction_window", 30))
  confidence_level = float(tf_config.get("confidence_level", 0.9))

  cloud = params.get("cloud", {})
  project_id = cloud.get("project_id")
  dataset_id = cloud.get("dataset_id")

  split_days = int(params.get("default", {}).get("split_train_and_test", 30))
  data_config = params.get("data", {})
  input_folder = data_config.get("clean_folder", "data/clean_data")
  in_csv = os.path.join(input_folder, f"{tag}.csv")

  if not os.path.exists(in_csv):
    logging.error(
        "Source cleaned dataset %s missing. Aborting TimesFM forecast.",
        in_csv,
    )
    return

  logging.info("Reading cleaned dataset from %s", in_csv)
  df = pd.read_csv(in_csv, parse_dates=[date_col]).sort_values(date_col)
  if df.empty or date_col not in df.columns:
    logging.error(
        "Dataset at %s is empty or lacks a '%s' column.", in_csv, date_col
    )
    return

  test_end_date = df[date_col].max()
  cutoff_date = (test_end_date - timedelta(days=split_days)).strftime(
      "%Y-%m-%d"
  )

  logging.info("Target context Tag: %s", tag)
  logging.info("Prediction window: %d", prediction_window)
  logging.info("Confidence level: %.2f", confidence_level)
  logging.info("Project ID: %s", project_id)
  logging.info("Dataset ID: %s", dataset_id)
  logging.info("Cutoff Date: %s", cutoff_date)

  if not project_id or not dataset_id or not cutoff_date:
    raise ValueError(
        "Missing mandatory BigQuery configuration attributes. Aborting TimesFM"
        " forecast."
    )

  solution_folder = os.path.join(
      data_config.get("forecast_folder", "data/forecast"), SOLUTION
  )
  os.makedirs(solution_folder, exist_ok=True)
  forecast_csv = os.path.join(solution_folder, f"{tag}_forecast.csv")

  # Establish client connection and query remotely
  client = bigquery.Client(project=project_id)
  query = build_timesfm_query(
      project_id,
      dataset_id,
      tag,
      cutoff_date,
      prediction_window,
      confidence_level,
      date_col,
      val_col,
  )

  logging.info("Dispatching TimesFM query mapping to BigQuery table: %s", tag)

  query_job = client.query(query)
  result = query_job.result().to_dataframe()

  logging.info("Exporting TimesFM forecast results to %s", forecast_csv)
  result.to_csv(forecast_csv, index=False)
  logging.info(
      "Successfully queried and exported TimesFM forecast strictly for tag: %s",
      tag,
  )

  utils.upload_to_gcs(local_filepath=forecast_csv)


if __name__ == "__main__":
  execute_timesfm_forecast()
