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

"""Tests for join_forecasts."""

import os
from unittest import mock

import pandas as pd
from pyfakefs import fake_filesystem_unittest

from pipelines import join_forecasts
from common import utils
from google3.testing.pybase import googletest


class JoinForecastsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_load_forecast_success(self):
    os.makedirs("data/generate_forecast/prophet", exist_ok=True)
    csv_path = "data/generate_forecast/prophet/test_tag_forecast.csv"
    self.fs.create_file(
        csv_path,
        contents="ds,yhat\n2026-01-01,100.0\n2026-01-02,200.0\n",
    )

    df = join_forecasts.load_forecast(csv_path, "ds", "yhat", "prophet")
    self.assertFalse(df.empty)
    self.assertIn("prophet", df.columns)
    self.assertLen(df, 2)
    self.assertEqual(df.index.name, "date")

  def test_load_forecast_missing_file(self):
    df = join_forecasts.load_forecast(
        "non_existent.csv", "ds", "yhat", "prophet"
    )
    self.assertTrue(df.empty)

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_aggregate_forecast_streams_success(self, mock_upload):
    os.makedirs("data/generate_forecast/prophet", exist_ok=True)
    os.makedirs("data/generate_forecast/timesfm", exist_ok=True)
    os.makedirs("data/generate_forecast/arima_plus", exist_ok=True)
    os.makedirs("data/collect_forecasts", exist_ok=True)

    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\n"
            "data:\n"
            "  generate_forecast_folder: data/generate_forecast\n"
            "  collect_forecasts_folder: data/collect_forecasts\n"
        ),
    )
    self.fs.create_file(
        "data/generate_forecast/prophet/test_tag_forecast.csv",
        contents="ds,yhat\n2026-01-01,100.0\n2026-01-02,200.0\n",
    )
    self.fs.create_file(
        "data/generate_forecast/timesfm/test_tag_forecast.csv",
        contents=(
            "forecast_timestamp,forecast_value\n"
            "2026-01-01,110.0\n2026-01-02,210.0\n"
        ),
    )
    self.fs.create_file(
        "data/generate_forecast/arima_plus/test_tag_forecast.csv",
        contents=(
            "forecast_timestamp,forecast_value\n"
            "2026-01-01,105.0\n2026-01-02,205.0\n"
        ),
    )

    join_forecasts.aggregate_forecast_streams("params.yaml")

    mock_upload.assert_called_once()
    out_csv = "data/collect_forecasts/test_tag.csv"
    self.assertTrue(os.path.exists(out_csv))
    merged = pd.read_csv(out_csv)
    self.assertIn("prophet", merged.columns)
    self.assertIn("timesfm", merged.columns)
    self.assertIn("arima_plus", merged.columns)
    self.assertIn("tag", merged.columns)

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_aggregate_forecast_streams_no_valid_outputs(self, mock_upload):
    self.fs.create_file("params.yaml", contents="tag: test_tag\n")
    with self.assertLogs(level="ERROR") as log:
      join_forecasts.aggregate_forecast_streams("params.yaml")
      self.assertIn("No valid outputs", log.records[0].getMessage())
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
