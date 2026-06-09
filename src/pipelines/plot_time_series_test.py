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

"""Tests for plot_time_series."""

import os
from unittest import mock

from pipelines import plot_time_series
from common import utils
from google3.testing.pybase import googletest
import pandas as pd
import plotly
from pyfakefs import fake_filesystem_unittest


class PlotTimeSeriesTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_validate_dataframe_success(self):
    df = pd.DataFrame({"date": ["2026-01-01"], "spend": [100.0]})
    plot_time_series._validate_dataframe(df, ["date", "spend"])

  def test_validate_dataframe_missing_columns(self):
    df = pd.DataFrame({"date": ["2026-01-01"]})
    with self.assertRaises(ValueError):
      plot_time_series._validate_dataframe(df, ["date", "spend"])

  def test_plot_ts_png_success(self):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
        "spend": [1000.0, 2000.0],
    })
    plot = plot_time_series.plot_ts_png(df, "TestBrand", "date", "spend")
    self.assertIsNotNone(plot)

  @mock.patch.object(plot_time_series.px, "line", autospec=True)
  def test_plot_ts_html_success(self, mock_px_line):
    mock_fig = mock.Mock()
    mock_px_line.return_value = mock_fig
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
        "spend": [1000.0, 2000.0],
    })
    fig = plot_time_series.plot_ts_html(df, "TestBrand", "date", "spend")
    self.assertEqual(fig, mock_fig)
    mock_fig.update_layout.assert_called_once()
    mock_fig.update_yaxes.assert_called_once()

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  @mock.patch.object(plot_time_series, "plot_ts_html", autospec=True)
  @mock.patch.object(plot_time_series.gg, "ggsave", autospec=True)
  def test_generate_plot_outputs_success(
      self, mock_ggsave, mock_plot_html, mock_upload
  ):
    mock_fig = mock.Mock()
    mock_plot_html.return_value = mock_fig
    os.makedirs("data/clean", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\n"
            "default:\n"
            "  date_column: date\n"
            "  prediction_column: spend\n"
            "data:\n"
            "  clean_folder: data/clean\n"
            "  plot_folder: plots/output\n"
        ),
    )
    self.fs.create_file(
        "data/clean/test_tag.csv",
        contents="date,spend\n2026-01-01,1500\n2026-02-01,2500\n",
    )

    plot_time_series.generate_plot_outputs("params.yaml")

    mock_ggsave.assert_called_once()
    mock_fig.write_html.assert_called_once()
    self.assertEqual(mock_upload.call_count, 2)

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_generate_plot_outputs_missing_source(self, mock_upload):
    self.fs.create_file(
        "params.yaml",
        contents="tag: test_tag\ndata:\n  clean_folder: data/non_existent\n",
    )
    with self.assertLogs(level="ERROR") as log:
      plot_time_series.generate_plot_outputs("params.yaml")
      self.assertIn("missing", log.records[0].getMessage())
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
