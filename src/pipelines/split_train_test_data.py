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

"""Splits historical temporal forecasting datasets into train and test segments.

Utilizes shared utility parameter parsing without relying on hardcoded defaults.
"""

from datetime import timedelta
import os
from typing import Tuple

from absl import logging
import pandas as pd

from common import utils


def split_dataframe(
    df: pd.DataFrame, split_days: int, date_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Splits a time series dataframe into train and test segments.

  Args:
      df: Dataframe containing configured date datetime column, sorted by date.
      split_days: Number of days from the end of the series allocated to test.
      date_col: Configured date column name.

  Returns:
      A tuple comprising (train_df, test_df).

  Raises:
      ValueError: If the dataframe is empty or missing the date_col column.
  """
  if date_col not in df.columns:
    raise ValueError(f"Input dataframe must contain a '{date_col}' column.")
  if df.empty:
    raise ValueError("Input dataframe cannot be empty.")

  test_end_date = df[date_col].max()
  test_start_date = test_end_date - timedelta(days=split_days)

  train_df = df[df[date_col] < test_start_date].copy()
  test_df = df[
      (df[date_col] >= test_start_date) & (df[date_col] <= test_end_date)
  ].copy()

  return train_df, test_df


def split_temporal_data(config_path: str = "params.yaml") -> None:
  """Main execution method dividing time series data into bounded splits.

  Args:
      config_path: Path to the YAML configuration file.

  Raises:
      ValueError: If split configuration parameters are invalid.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)
  date_col = utils.get_date_col(params)

  # Unpack split logic parameter
  split_days = params.get("default", {}).get("split_train_and_test", 30)
  if not isinstance(split_days, int) or split_days <= 0:
    raise ValueError(
        "Mandatory configuration 'default.split_train_and_test' must be a"
        " positive integer."
    )

  data_config = params.get("data", {})
  input_folder = data_config.get("clean_folder", "data/clean_data")
  in_csv = os.path.join(input_folder, f"{tag}.csv")

  if not os.path.exists(in_csv):
    logging.error("Source dataset file %s missing. Aborting split.", in_csv)
    return

  logging.info("Reading cleaned dataset from %s", in_csv)
  df = pd.read_csv(in_csv, parse_dates=[date_col]).sort_values(date_col)

  train_df, test_df = split_dataframe(df, split_days, date_col)

  test_end_date = df[date_col].max()
  if test_df[date_col].max() != test_end_date:
    logging.warning("Test upper bound mismatch detected for tag %s", tag)

  out_dir = data_config.get("split_folder", "data/split_data")
  os.makedirs(out_dir, exist_ok=True)

  train_csv = os.path.join(out_dir, f"{tag}_train.csv")
  test_csv = os.path.join(out_dir, f"{tag}_test.csv")

  logging.info("Saving training split to %s", train_csv)
  train_df.to_csv(train_csv, index=False)
  logging.info("Saving test split to %s", test_csv)
  test_df.to_csv(test_csv, index=False)

  utils.upload_to_gcs(local_filepath=train_csv)
  utils.upload_to_gcs(local_filepath=test_csv)
  logging.info(
      "Successfully split mapped dataset fully dedicated to tag: %s", tag
  )


if __name__ == "__main__":
  split_temporal_data()
