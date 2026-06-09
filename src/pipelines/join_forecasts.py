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

"""Consolidates individual algorithm outputs into unified comparative predictive sequences.

Applies shared configuration conventions adapting dynamic tag contexts without
hardcoded mapping tuples.
"""

import os

from absl import logging
import pandas as pd

from common import utils


def load_forecast(
    path: str, date_col: str, val_col: str, model_name: str
) -> pd.DataFrame:
  """Safely parses target predictive datasets aligning specific index representations.

  Args:
      path: Target source file descriptor.
      date_col: Timestamp header label.
      val_col: Absolute predictive value header.
      model_name: Standard key to assign within output array.

  Returns:
      Normalized index-aligned Pandas DataFrame.
  """
  if not os.path.exists(path):
    logging.warning("Expected source forecast mapping %s not found.", path)
    return pd.DataFrame()

  df = pd.read_csv(
      path,
      usecols=[date_col, val_col],
      parse_dates=[date_col],
  ).rename(columns={date_col: "date", val_col: model_name})

  if df["date"].dt.tz is not None:
    df["date"] = df["date"].dt.tz_localize(None)

  return df.set_index("date")


def aggregate_forecast_streams(config_path: str = "params.yaml") -> None:
  """Main process method consolidating multi-solution metrics based exclusively on specific target tags.

  Args:
      config_path: Path to the YAML configuration file.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)

  data_config = params.get("data", {})
  gen_folder = data_config.get(
      "generate_forecast_folder", "data/generate_forecast"
  )

  # Build specific expected model paths securely mapping execution tag
  prophet_csv = os.path.join(gen_folder, "prophet", f"{tag}_forecast.csv")
  tfm_csv = os.path.join(gen_folder, "timesfm", f"{tag}_forecast.csv")
  arp_csv = os.path.join(gen_folder, "arima_plus", f"{tag}_forecast.csv")

  logging.info("Consolidating forecast streams targeting Tag: %s", tag)

  # Load corresponding normalized outcomes
  prophet_df = load_forecast(prophet_csv, "ds", "yhat", "prophet")
  tfm_df = load_forecast(
      tfm_csv, "forecast_timestamp", "forecast_value", "timesfm"
  )
  arp_df = load_forecast(
      arp_csv, "forecast_timestamp", "forecast_value", "arima_plus"
  )

  dfs_to_concat = [df for df in [prophet_df, tfm_df, arp_df] if not df.empty]

  if not dfs_to_concat:
    logging.error(
        "No valid outputs available to merge for tag %s. Aborting merge.", tag
    )
    return

  merged = pd.concat(dfs_to_concat, axis=1)
  merged["tag"] = tag

  out_dir = data_config.get(
      "collect_forecasts_folder", "data/collect_forecasts"
  )
  os.makedirs(out_dir, exist_ok=True)
  out_csv = os.path.join(out_dir, f"{tag}.csv")
  merged.to_csv(out_csv)
  logging.info("Unified predictive dashboard complete. Saved to: %s", out_csv)

  utils.upload_to_gcs(local_filepath=out_csv)


if __name__ == "__main__":
  aggregate_forecast_streams()
