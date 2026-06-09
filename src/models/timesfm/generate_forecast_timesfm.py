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

"""Generates long-term projections querying continuous TimesFM models via BigQuery.

Applies shared configuration and active execution tag discovery removing
hardcoded segment arrays.
"""

import os

from absl import logging
from google.cloud import bigquery
import pandas as pd

from common import utils

SOLUTION = "timesfm"


def build_generate_timesfm_query(
    project_id: str,
    dataset_id: str,
    table_id: str,
    start_date: str,
    horizon: int,
    confidence_level: float,
    date_col: str,
    val_col: str,
) -> str:
  """Constructs the BigQuery BQML TimesFM long-term generation query string.

  Args:
      project_id: Target GCP project identifier.
      dataset_id: Target BigQuery dataset identifier.
      table_id: Source table name.
      start_date: Upper bound date filtering training data.
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
                WHERE {date_col} < '{start_date}'
            ),
            horizon => {horizon},
            confidence_level => {confidence_level},
            timestamp_col => '{date_col}',
            data_col => '{val_col}'
        );
    """


def generate_final_timesfm_sequence(config_path: str = "params.yaml") -> None:
  """Primary execution method formatting remote queries and processing predictive boundaries.

  Args:
      config_path: Path to the YAML configuration file.

  Raises:
      ValueError: If mandatory configuration parameters or boundaries are
        missing.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)
  date_col = utils.get_date_col(params)
  val_col = utils.get_val_col(params)

  tf_config = params.get("forecast_timesfm", {})
  confidence_level = float(tf_config.get("confidence_level", 0.9))

  cloud = params.get("cloud", {})
  project_id = cloud.get("project_id")
  dataset_id = cloud.get("dataset_id")

  gen_config = params.get("generate_forecast", {})
  start_date = gen_config.get("start_date")
  end_date = gen_config.get("end_date")

  data_config = params.get("data", {})
  input_folder = data_config.get("split_folder", "data/split_data")
  train_csv = os.path.join(input_folder, f"{tag}_train.csv")

  if not os.path.exists(train_csv):
    logging.error(
        "Source training split %s missing. Aborting TimesFM generation.",
        train_csv,
    )
    return

  logging.info("Reading training split from %s", train_csv)
  df = pd.read_csv(train_csv, parse_dates=[date_col]).sort_values(date_col)
  if df.empty or date_col not in df.columns:
    logging.error(
        "Dataset at %s is empty or lacks a '%s' column.", train_csv, date_col
    )
    return

  train_end_date = df[date_col].max()

  logging.info("Target context Tag: %s", tag)
  logging.info("Confidence level: %.2f", confidence_level)
  logging.info("Project ID: %s", project_id)
  logging.info("Dataset ID: %s", dataset_id)
  logging.info("Start Date: %s", start_date)
  logging.info("End Date: %s", end_date)

  if not all([project_id, dataset_id, start_date, end_date]):
    raise ValueError(
        "Missing essential mandatory configuration parameter attributes."
        " Aborting TimesFM generation."
    )

  start_dt = pd.to_datetime(start_date)
  end_dt = pd.to_datetime(end_date)
  horizon = (end_dt - train_end_date).days
  logging.info("Calculated projection horizon: %d days", horizon)

  out_dir = os.path.join(
      data_config.get("generate_forecast_folder", "data/generate_forecast"),
      SOLUTION,
  )
  os.makedirs(out_dir, exist_ok=True)
  output_csv = os.path.join(out_dir, f"{tag}_forecast.csv")

  client = bigquery.Client(project=project_id)
  query = build_generate_timesfm_query(
      project_id,
      dataset_id,
      tag,
      start_date,
      horizon,
      confidence_level,
      date_col,
      val_col,
  )

  logging.info(
      "Dispatching TimesFM generation query corresponding directly to BigQuery"
      " table: %s",
      tag,
  )

  query_job = client.query(query)
  result = query_job.result().to_dataframe()

  # Normalize date representation ensuring seamless interval comparisons
  if "forecast_timestamp" in result.columns:
    date_col = "forecast_timestamp"
  elif "date" in result.columns:
    date_col = "date"
  else:
    date_col = result.columns[0]

  result[date_col] = (
      pd.to_datetime(result[date_col]).dt.tz_localize(None).dt.normalize()
  )

  filtered_result = result[
      (result[date_col] >= start_dt) & (result[date_col] <= end_dt)
  ]

  logging.info(
      "Saving filtered long-term BQML generated sequence to %s", output_csv
  )
  filtered_result.to_csv(output_csv, index=False)
  logging.info(
      "Successfully filtered long-term BQML generated sequence safely saved"
      " to: %s",
      output_csv,
  )

  utils.upload_to_gcs(local_filepath=output_csv)


if __name__ == "__main__":
  generate_final_timesfm_sequence()
