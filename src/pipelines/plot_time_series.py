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

"""Visualizes temporal forecasting time series using shared utilities.

This module provides functionality to generate both static line charts (PNG)
via Plotnine and interactive web layouts (HTML) via Plotly.
"""

import os
from typing import Dict

from absl import logging
from mizani.formatters import label_currency
from mizani.formatters import label_date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotnine as gg

from common import utils


def _validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
  """Validates that the dataframe contains all required columns."""
  missing = [col for col in required_columns if col not in df.columns]
  if missing:
    raise ValueError(f"Input dataframe is missing required columns: {missing}")


def plot_ts_png(
    df: pd.DataFrame, brand: str, date_col: str, val_col: str
) -> gg.ggplot:
  """Generates a static line chart via Plotnine.

  Args:
      df: Dataframe containing configured date and prediction value columns.
      brand: Brand or target tag identifier used in the plot title.
      date_col: Configured date column name.
      val_col: Configured prediction value column name.

  Returns:
      A Plotnine ggplot object representing the static line chart.

  Raises:
      ValueError: If required columns are missing from the dataframe.
  """
  _validate_dataframe(df, [date_col, val_col])
  gcolours: Dict[str, str] = {"blue": "#1a73e8", "green": "#1e8e3e"}
  return (
      gg.ggplot(df, gg.aes(date_col, val_col))
      + gg.geom_line(colour=gcolours["blue"])
      + gg.scale_y_continuous(labels=label_currency(precision=0, big_mark=","))
      + gg.scale_x_date(labels=label_date(fmt="%b %Y"))
      + gg.ggtitle(f"{val_col.capitalize()}: {brand}")
      + gg.theme_minimal()
      + gg.theme(axis_title=gg.element_blank())
  )


def plot_ts_html(
    df: pd.DataFrame, brand: str, date_col: str, val_col: str
) -> go.Figure:
  """Constructs an interactive HTML plot layout via Plotly Express.

  Args:
      df: Dataframe containing configured date and prediction value columns.
      brand: Brand or target tag identifier used in the plot title.
      date_col: Configured date column name.
      val_col: Configured prediction value column name.

  Returns:
      A Plotly Figure object representing the interactive chart.

  Raises:
      ValueError: If required columns are missing from the dataframe.
  """
  _validate_dataframe(df, [date_col, val_col])
  fig = px.line(
      df, x=date_col, y=val_col, title=f"{val_col.capitalize()}: {brand}"
  )
  fig.update_layout(xaxis_title=None, yaxis_title=None)
  fig.update_yaxes(tickprefix="$")
  return fig


def generate_plot_outputs(config_path: str = "params.yaml") -> None:
  """Executes the main plotting pipeline and exports visual artifacts.

  Args:
      config_path: Path to the YAML configuration file.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)
  date_col = utils.get_date_col(params)
  val_col = utils.get_val_col(params)

  data_config = params.get("data", {})
  input_folder = data_config.get("clean_folder", "data/clean_data")
  in_csv = os.path.join(input_folder, f"{tag}.csv")

  out_folder = data_config.get("plot_folder", "plots/plot_time_series")
  os.makedirs(out_folder, exist_ok=True)

  if not os.path.exists(in_csv):
    logging.error("Source file %s missing. Aborting plot generation.", in_csv)
    return

  logging.info("Reading cleaned dataset from %s", in_csv)
  df = pd.read_csv(in_csv, parse_dates=[date_col])

  png_path = os.path.join(out_folder, f"{tag}.png")
  logging.info("Generating static PNG plot at %s", png_path)
  png_plot = plot_ts_png(df, tag, date_col, val_col)
  gg.ggsave(png_plot, filename=png_path, width=12, height=5, units="in")

  html_path = os.path.join(out_folder, f"{tag}.html")
  logging.info("Generating interactive HTML plot at %s", html_path)
  html_plot = plot_ts_html(df, tag, date_col, val_col)
  html_plot.write_html(html_path)

  utils.upload_to_gcs(local_filepath=png_path)
  utils.upload_to_gcs(local_filepath=html_path)


if __name__ == "__main__":
  generate_plot_outputs()
