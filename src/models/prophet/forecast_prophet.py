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

"""Constructs dynamic predictive time series models optimized via Optuna.

Implements deferred parameter loading avoiding module-level side effects.
"""

import os
from typing import Any, Dict

from absl import logging
import numpy as np
import optuna
import pandas as pd
import prophet
from prophet import diagnostics
from prophet import serialize

from common import utils

Prophet = prophet.Prophet
model_to_json = serialize.model_to_json
cross_validation = diagnostics.cross_validation

SOLUTION = "prophet"


def _validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
  """Validates that the dataframe contains all required columns."""
  missing = [col for col in required_columns if col not in df.columns]
  if missing:
    raise ValueError(f"Input dataframe is missing required columns: {missing}")


def objective(
    trial: optuna.Trial, df: pd.DataFrame, config: Dict[str, Any]
) -> float:
  """Defines the Optuna objective function for Prophet hyperparameter tuning.

  Trains a Prophet model with suggested hyperparameters and calculates absolute
  difference between actual and predicted values across validation cross folds.

  Args:
      trial: Configured Optuna trial object.
      df: The training DataFrame holding 'ds' and 'y' series.
      config: Optimization boundary definitions dictionary.

  Returns:
      Absolute sum discrepancy between actual and predicted cross-validation
      arrays.
  """
  parameters = {
      "changepoint_prior_scale": trial.suggest_float(
          "changepoint_prior_scale",
          config.get("cp_prior_min", 0.05),
          config.get("cp_prior_max", 5.0),
      ),
      "changepoint_range": trial.suggest_float(
          "changepoint_range",
          config.get("cp_range_min", 0.1),
          config.get("cp_range_max", 0.9),
      ),
      "seasonality_mode": trial.suggest_categorical(
          "seasonality_mode", config.get("ssn_mode", ["multiplicative"])
      ),
      "yearly_seasonality": trial.suggest_categorical(
          "yearly_seasonality", [True]
      ),
      "weekly_seasonality": trial.suggest_categorical(
          "weekly_seasonality", [True]
      ),
      "seasonality_prior_scale": trial.suggest_float(
          "seasonality_prior_scale",
          config.get("ssn_prior_min", 0.05),
          config.get("ssn_prior_max", 5.0),
      ),
      "holidays_prior_scale": trial.suggest_float(
          "holidays_prior_scale",
          config.get("holidays_prior_min", 0.05),
          config.get("holidays_prior_max", 5.0),
      ),
  }

  model = Prophet(**parameters, interval_width=0.9)
  model.add_country_holidays(country_name=config.get("country_name", "AU"))
  model.fit(df)

  # Check if parallel execution is overridden (e.g., for single-core or tests)
  cv_parallel = config.get("cv_parallel", "processes")
  cv_parallel_arg = None if cv_parallel == "None" else cv_parallel

  cv_results = cross_validation(
      model=model,
      initial=config.get("initial", "400 days"),
      period=config.get("period", "30 days"),
      horizon=config.get("horizon", "30 days"),
      parallel=cv_parallel_arg,
  )

  # cv_results['y'] and cv_results['yhat'] are in log1p space, back-transform
  # them
  actual_sum = np.expm1(cv_results["y"]).sum()
  forecast_sum = np.expm1(cv_results["yhat"]).sum()
  return abs(actual_sum - forecast_sum)


def run_prophet_training_with_optimization(
    config_path: str = "params.yaml",
) -> None:
  """Main sequential execution method handling configuration and optimization loops.

  Args:
      config_path: Path to the YAML configuration file.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)
  date_col = utils.get_date_col(params)
  val_col = utils.get_val_col(params)

  prophet_hyper = params.get("prophet_hyper", {})
  fit_model = params.get("fit_model", {})

  # Aggregate optimization variables into a single accessible configuration map
  opt_config = {
      **prophet_hyper,
      "country_name": params.get("default", {}).get("country_name", "AU"),
      "initial": fit_model.get("initial", "400 days"),
      "period": fit_model.get("period", "30 days"),
      "horizon": fit_model.get("horizon", "30 days"),
      "cv_parallel": fit_model.get("cv_parallel", "processes"),
  }

  data_config = params.get("data", {})
  input_folder = data_config.get("split_folder", "data/split_data")
  train_csv = os.path.join(input_folder, f"{tag}_train.csv")

  if not os.path.exists(train_csv):
    logging.error(
        "Training dataset %s missing. Aborting optimization.", train_csv
    )
    return

  logging.info("Reading training split from %s", train_csv)
  train_df = pd.read_csv(train_csv, parse_dates=[date_col])
  _validate_dataframe(train_df, [date_col, val_col])

  prophet_df = train_df.rename(columns={date_col: "ds", val_col: "y"})
  prophet_df["y"] = np.log1p(prophet_df["y"])

  if prophet_df.empty:
    logging.warning(
        "Skipping %s: Training dataframe is empty at %s", tag, train_csv
    )
    return

  logging.info("Executing Optuna hyperparameter study for %s", tag)
  study = optuna.create_study(direction="minimize")
  study.optimize(
      lambda trial: objective(trial, prophet_df, opt_config),
      n_trials=fit_model.get("trial_num", 20),
      n_jobs=1,
  )

  best_params_config = study.best_params
  logging.info("Optimal parameters discovered: %s", best_params_config)

  # Final global optimal model build
  optimal_model = Prophet(**best_params_config, interval_width=0.9)
  optimal_model.add_country_holidays(
      country_name=opt_config.get("country_name", "AU")
  )
  optimal_model.fit(prophet_df)

  solution_folder = os.path.join(
      data_config.get("forecast_folder", "data/forecast"), SOLUTION
  )
  os.makedirs(solution_folder, exist_ok=True)

  model_json = os.path.join(solution_folder, f"{tag}_model.json")
  logging.info("Serializing optimal model to %s", model_json)
  with open(model_json, "w") as fout:
    fout.write(model_to_json(optimal_model))

  # Export future projections
  test_csv = os.path.join(input_folder, f"{tag}_test.csv")
  if not os.path.exists(test_csv):
    logging.error("Test dataset %s missing. Aborting projections.", test_csv)
    return

  test_df = pd.read_csv(test_csv, parse_dates=[date_col])
  _validate_dataframe(test_df, [date_col])

  future_dates = pd.date_range(
      start=test_df[date_col].min(), end=test_df[date_col].max(), freq="D"
  )

  future = pd.DataFrame({"ds": future_dates})
  forecast = optimal_model.predict(future)

  # Back-transform forecast and bounds ensuring strictly positive values
  for col in ["yhat", "yhat_lower", "yhat_upper"]:
    forecast[col] = np.clip(np.expm1(forecast[col]), 0.0, None)

  forecast_csv = os.path.join(solution_folder, f"{tag}_forecast.csv")
  logging.info("Saving future projections to %s", forecast_csv)
  forecast.to_csv(forecast_csv, index=False)

  output_plot = os.path.join(solution_folder, f"{tag}_plot.png")
  logging.info("Saving forecast plot to %s", output_plot)
  fig = optimal_model.plot(forecast)
  fig.savefig(output_plot)

  utils.upload_to_gcs(local_filepath=model_json)
  utils.upload_to_gcs(local_filepath=forecast_csv)
  utils.upload_to_gcs(local_filepath=output_plot)
  logging.info(
      "Saved optimization trials, model to %s, and outcomes exclusively for:"
      " %s",
      model_json,
      tag,
  )


if __name__ == "__main__":
  run_prophet_training_with_optimization()
