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

"""Preprocesses raw temporal forecasting datasets using shared utilities.

This module provides functionality to load raw time-series data, apply
configured query filters, and export the cleaned data to both local storage
and Google Cloud Storage (GCS).
"""

import os
from typing import Any, Dict

from absl import logging
from common import utils
import pandas as pd


def preprocess_dataframe(
    df: pd.DataFrame, data_filter: str, date_col: str
) -> pd.DataFrame:
  """Cleans and filters the raw forecasting dataframe.

  Args:
    df: Raw dataframe containing configured date column.
    data_filter: Query string used to filter the dataframe.
    date_col: Configured date column name.

  Returns:
    Processed dataframe with date_col as the index and filtered rows.

  Raises:
    ValueError: If the input dataframe is missing the date_col column.
  """
  if date_col not in df.columns:
    raise ValueError(f"Input dataframe must contain a '{date_col}' column.")

  # Ensure date_col is datetime type if not already parsed
  df[date_col] = pd.to_datetime(df[date_col])

  df_processed = df.set_index(date_col)

  logging.info(
      "Dataset date range before filtering: %s to %s",
      df_processed.index.min(),
      df_processed.index.max(),
  )

  filtered_df = df_processed.query(data_filter)
  logging.info(
      "Filtered dataset from %d to %d rows.",
      len(df_processed),
      len(filtered_df),
  )
  return filtered_df


def clean_and_export_data(config_path: str = "params.yaml") -> None:
  """Executes the main data cleaning pipeline and exports artifacts.

  Args:
    config_path: Path to the YAML configuration file.

  Raises:
    KeyError: If mandatory configuration parameters are missing.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)
  date_col = utils.get_date_col(params)

  data_filter = params.get("data_filter")
  if not data_filter:
    raise KeyError(
        "Mandatory configuration parameter 'data_filter' is missing."
    )

  data_config: Dict[str, Any] = params.get("data", {})
  data_csv = data_config.get("local_file", "data/raw/sem_ts_gfs.csv")
  out_dir = data_config.get("clean_folder", "data/clean_data")
  os.makedirs(out_dir, exist_ok=True)

  clean_csv = os.path.join(out_dir, f"{tag}.csv")

  if not os.path.exists(data_csv):
    logging.error("Source file %s missing. Aborting data cleaning.", data_csv)
    return

  logging.info("Reading raw dataset from %s", data_csv)
  df_raw = pd.read_csv(data_csv, parse_dates=[date_col])

  filtered_df = preprocess_dataframe(df_raw, data_filter, date_col)

  logging.info("Saving cleaned dataset to %s", clean_csv)
  filtered_df.to_csv(clean_csv)
  logging.info("Successfully cleaned dataset mapped exactly to: %s", clean_csv)

  utils.upload_to_gcs(local_filepath=clean_csv)


if __name__ == "__main__":
  clean_and_export_data()
