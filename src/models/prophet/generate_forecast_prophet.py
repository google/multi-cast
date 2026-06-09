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

"""Generates long-term predictions loading serialized Prophet stubs.

Applies shared configuration and active target tag discovery completely removing
specific segment mappings.
"""

import os

from absl import logging
import numpy as np
import pandas as pd
import prophet
from prophet import serialize

from common import utils

Prophet = prophet.Prophet
model_from_json = serialize.model_from_json

SOLUTION = "prophet"


def prepare_forecasting_data(
    csv_path: str, date_col: str, val_col: str
) -> pd.DataFrame:
  """Loads reference datasets and aligns dimensions for predictive pipelines.

  Args:
      csv_path: Path pointing directly to target cleaned data source.
      date_col: Configured date column name.
      val_col: Configured prediction value column name.

  Returns:
      A Pandas DataFrame holding renamed attributes matching Prophet format
      requirements.

  Raises:
      ValueError: If the dataset is missing required date or prediction columns.
  """
  if not os.path.exists(csv_path):
    logging.warning("Reference dataset path %s does not exist.", csv_path)
    return pd.DataFrame()

  df = pd.read_csv(csv_path, parse_dates=[date_col])
  if date_col not in df.columns or val_col not in df.columns:
    raise ValueError(
        f"Dataset at {csv_path} lacks required '{date_col}' or '{val_col}'"
        " columns."
    )

  return df.rename(columns={date_col: "ds", val_col: "y"})


def generate_final_sequence(config_path: str = "params.yaml") -> None:
  """Primary method executing model deserialization and producing output arrays.

  Args:
      config_path: Path to the YAML configuration file.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)
  date_col = utils.get_date_col(params)
  val_col = utils.get_val_col(params)

  data_config = params.get("data", {})
  solution_folder = os.path.join(
      data_config.get("forecast_folder", "data/forecast"), SOLUTION
  )
  model_json = os.path.join(solution_folder, f"{tag}_model.json")
  clean_csv = os.path.join(
      data_config.get("clean_folder", "data/clean_data"), f"{tag}.csv"
  )
  out_dir = os.path.join(
      data_config.get("generate_forecast_folder", "data/generate_forecast"),
      SOLUTION,
  )
  os.makedirs(out_dir, exist_ok=True)
  forecast_csv = os.path.join(out_dir, f"{tag}_forecast.csv")

  if not os.path.exists(model_json) or not os.path.exists(clean_csv):
    logging.error(
        "Missing baseline model %s or input mapping %s for tag %s. Aborting"
        " generation.",
        model_json,
        clean_csv,
        tag,
    )
    return

  # Extract baseline hyperparameters safely from previously serialized
  # representations
  logging.info(
      "Reconstructing Prophet framework strictly based on tag: %s", tag
  )
  with open(model_json, "r") as f:
    pretrained_model = model_from_json(f.read())

  best_params = {
      "changepoint_prior_scale": pretrained_model.changepoint_prior_scale,
      "changepoint_range": pretrained_model.changepoint_range,
      "seasonality_mode": pretrained_model.seasonality_mode,
      "seasonality_prior_scale": pretrained_model.seasonality_prior_scale,
      "holidays_prior_scale": pretrained_model.holidays_prior_scale,
      "yearly_seasonality": True,
      "weekly_seasonality": True,
  }

  df = prepare_forecasting_data(clean_csv, date_col, val_col)
  if df.empty:
    logging.warning("Input dataset corresponding to %s is empty.", clean_csv)
    return

  # Log-transform target variable
  df["y"] = np.log1p(df["y"])

  model = Prophet(**best_params, interval_width=0.9)
  country_name = params.get("default", {}).get("country_name", "AU")
  model.add_country_holidays(country_name=country_name)
  model.fit(df)

  # Bound future sequence arrays mapped precisely to parameters
  generate_forecast = params.get("generate_forecast", {})
  start_date = generate_forecast.get("start_date")
  end_date = generate_forecast.get("end_date")

  if not start_date or not end_date:
    logging.error(
        "Mandatory forecasting boundaries 'start_date' or 'end_date' missing"
        " from configuration."
    )
    return

  logging.info(
      "Generating future date sequences from %s to %s", start_date, end_date
  )
  future_dates = pd.date_range(start=start_date, end=end_date, freq="D")
  future = pd.DataFrame({"ds": future_dates})
  forecast = model.predict(future)

  # Back-transform forecast and bounds ensuring strictly positive values
  for col in ["yhat", "yhat_lower", "yhat_upper"]:
    forecast[col] = np.clip(np.expm1(forecast[col]), 0.0, None)

  # Export clean resulting projections
  logging.info("Saving continuous baseline forecasts to %s", forecast_csv)
  forecast.to_csv(forecast_csv, index=False)
  logging.info(
      "Successfully mapped continuous baseline forecasts and saved precisely"
      " to: %s",
      forecast_csv,
  )

  utils.upload_to_gcs(local_filepath=forecast_csv)


if __name__ == "__main__":
  generate_final_sequence()
