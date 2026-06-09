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

"""Validates TimesFM predictive segment arrays tracking absolute accuracy metrics.

Calculates MAPE and deviation metrics storing target outputs dynamically.
"""

import json
import os
from typing import Dict

from absl import logging
from mizani.labels import label_currency
from mizani.labels import label_date
import numpy as np
import pandas as pd
import plotnine as gg
from sklearn.metrics import mean_absolute_percentage_error as mape

from common import utils

SOLUTION = "timesfm"


def calculate_metrics(
    actual_series: pd.Series, forecast_series: pd.Series
) -> Dict[str, float]:
  """Computes evaluation accuracy metrics between actuals and predictions.

  Args:
      actual_series: Ground truth actual values.
      forecast_series: Predicted forecast values.

  Returns:
      Dictionary comprising 'mape', 'actual_sum', 'forecast_sum', and
      'deviation'.

  Raises:
      ValueError: If actual sum is zero preventing division by zero.
  """
  calc_actual_sum = actual_series.sum()
  if calc_actual_sum == 0:
    raise ValueError("Actual sum is zero; cannot compute deviation ratio.")

  calc_mape = mape(y_true=actual_series, y_pred=forecast_series)
  calc_forecast_sum = forecast_series.sum()
  calc_dev = float(
      np.abs(calc_actual_sum - calc_forecast_sum) / calc_actual_sum
  )

  return {
      "mape": float(calc_mape),
      "actual_sum": float(calc_actual_sum),
      "forecast_sum": float(calc_forecast_sum),
      "deviation": calc_dev,
  }


def generate_ribbon_plot(
    plot_data: pd.DataFrame,
    min_test_date: pd.Timestamp,
    tag: str,
    mape_val: float,
) -> gg.ggplot:
  """Constructs a comparative Plotnine ribbon visualization curve for TimesFM.

  Args:
      plot_data: Dataframe holding actual, tfm_forecast, tfm_lower_bound,
        tfm_upper_bound.
      min_test_date: Boundary date separating train and test segments.
      tag: Target tag identifier used in the plot title.
      mape_val: Calculated MAPE score displayed in the title.

  Returns:
      A Plotnine ggplot object representing the comparative ribbon curve.
  """
  gcolours: Dict[str, str] = {
      "blue": "#1a73e8",
      "green": "#1e8e3e",
      "red": "#d93025",
  }

  p = (
      gg.ggplot(plot_data, gg.aes(x="date"))
      + gg.geom_point(gg.aes(y="actual", color="'Actual'"))
      + gg.geom_line(gg.aes(y="actual", color="'Actual'"))
      + gg.geom_line(
          gg.aes(y="tfm_forecast", color="'TimesFM Forecast'"), size=1
      )
      + gg.geom_ribbon(
          gg.aes(ymin="tfm_lower_bound", ymax="tfm_upper_bound"),
          fill=gcolours["blue"],
          alpha=0.2,
      )
      + gg.geom_vline(
          xintercept=min_test_date, color="red", linetype="dashed", size=1
      )
      + gg.theme_bw()
      + gg.scale_x_date(labels=label_date("%b %Y"))
      + gg.scale_y_continuous(labels=label_currency(precision=0, big_mark=","))
      + gg.scale_color_manual(
          values={"Actual": "black", "TimesFM Forecast": gcolours["blue"]},
      )
      + gg.theme(
          axis_title=gg.element_blank(),
          legend_title=gg.element_blank(),
          legend_position="top",
          figure_size=(10, 5),
      )
      + gg.ggtitle(f"Spend - {tag} (MAPE: {mape_val:.3f})")
  )
  return p


def evaluate_tfm_forecast_precision_and_plot(
    config_path: str = "params.yaml",
) -> None:
  """Main evaluation process mapping target tags and calculating TimesFM performance metrics.

  Args:
      config_path: Path to the YAML configuration file.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)
  date_col = utils.get_date_col(params)
  val_col = utils.get_val_col(params)

  data_config = params.get("data", {})
  input_folder = data_config.get("split_folder", "data/split_data")
  train_csv = os.path.join(input_folder, f"{tag}_train.csv")
  test_csv = os.path.join(input_folder, f"{tag}_test.csv")

  solution_folder = os.path.join(
      data_config.get("forecast_folder", "data/forecast"), SOLUTION
  )
  os.makedirs(solution_folder, exist_ok=True)
  forecast_csv = os.path.join(solution_folder, f"{tag}_forecast.csv")

  if not all(os.path.exists(p) for p in [train_csv, test_csv, forecast_csv]):
    logging.error(
        "Missing prerequisite mapping files for tag %s under TimesFM."
        " Aborting evaluation.",
        tag,
    )
    return

  out_dir = os.path.join(
      data_config.get("evaluate_folder", "data/evaluate"), SOLUTION
  )
  os.makedirs(out_dir, exist_ok=True)
  metrics_json = os.path.join(out_dir, f"{tag}_metrics.json")
  output_plot = os.path.join(out_dir, f"{tag}_plot.png")

  # Read actual sub-segments
  logging.info("Reading actuals and TimesFM forecast datasets for %s", tag)
  train_df = pd.read_csv(train_csv, parse_dates=[date_col])
  test_df = pd.read_csv(test_csv, parse_dates=[date_col])
  min_test_date = test_df[date_col].min()

  actual_df = pd.concat([train_df, test_df], ignore_index=True).rename(
      columns={date_col: "date", val_col: "actual"}
  )

  # Parse predictive outputs safely
  forecast_df = pd.read_csv(
      forecast_csv, parse_dates=["forecast_timestamp"]
  ).rename(
      columns={
          "forecast_timestamp": "date",
          "forecast_value": "tfm_forecast",
          "prediction_interval_lower_bound": "tfm_lower_bound",
          "prediction_interval_upper_bound": "tfm_upper_bound",
      }
  )
  forecast_df["date"] = forecast_df["date"].dt.tz_localize(None)

  joined_df = pd.merge(actual_df, forecast_df, on=["date"], how="left")[[
      "date",
      "actual",
      "tfm_forecast",
      "tfm_lower_bound",
      "tfm_upper_bound",
  ]]

  joined_df.loc[joined_df["date"] < min_test_date, "tfm_forecast"] = np.nan
  joined_df.loc[joined_df["date"] < min_test_date, "tfm_lower_bound"] = np.nan
  joined_df.loc[joined_df["date"] < min_test_date, "tfm_upper_bound"] = np.nan

  joined_test = joined_df[joined_df["date"] >= min_test_date]

  # Calculations
  joined_clean = joined_test.dropna(subset=["actual", "tfm_forecast"])
  if joined_clean.empty:
    logging.warning("No overlapping clean evaluation records found for %s", tag)
    return

  metrics = calculate_metrics(
      joined_clean["actual"], joined_clean["tfm_forecast"]
  )
  logging.info("TIMESFM %s MAPE: %.3f", tag, metrics["mape"])
  logging.info("TIMESFM %s Actual Sum: %.3f", tag, metrics["actual_sum"])
  logging.info("TIMESFM %s Forecast Sum: %.3f", tag, metrics["forecast_sum"])
  logging.info("TIMESFM %s Deviation: %.3f", tag, metrics["deviation"])

  logging.info("Writing TimesFM evaluation metrics to %s", metrics_json)
  with open(metrics_json, "w") as f:
    json.dump(metrics, f)

  # Render comparative visualization
  plot_data = joined_df.copy()
  plot_data["date"] = pd.to_datetime(plot_data["date"])
  cutoff_date = plot_data["date"].max() - pd.DateOffset(days=120)
  plot_data = plot_data[plot_data["date"] >= cutoff_date]

  logging.info("Generating TimesFM evaluation ribbon plot at %s", output_plot)
  p = generate_ribbon_plot(plot_data, min_test_date, tag, metrics["mape"])
  p.save(output_plot)

  utils.upload_to_gcs(local_filepath=metrics_json)
  utils.upload_to_gcs(local_filepath=output_plot)
  logging.info("TimesFM evaluation completed successfully for tag: %s", tag)


if __name__ == "__main__":
  evaluate_tfm_forecast_precision_and_plot()
