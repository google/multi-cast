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

"""Tests for forecast_arima_plus."""

import os
from unittest import mock

from google.cloud import bigquery
import pandas as pd
from pyfakefs import fake_filesystem_unittest

from common import utils
from models.arima_plus import forecast_arima_plus
from google3.testing.pybase import googletest


class ForecastArimaPlusTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_build_arima_plus_query_success(self):
    query = forecast_arima_plus.build_arima_plus_query(
        "proj", "dataset", "table", "2026-01-01", 30, 0.95, "date", "spend"
    )
    self.assertIn("proj.dataset.table_arima_plus", query)
    self.assertIn("2026-01-01", query)
    self.assertIn("horizon, 0.95", query)

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  @mock.patch.object(bigquery, "Client", autospec=True)
  def test_execute_arima_plus_forecast_success(
      self, mock_client_cls, mock_upload
  ):
    mock_client = mock.Mock()
    mock_client_cls.return_value = mock_client
    mock_job = mock.Mock()
    mock_client.query.return_value = mock_job
    mock_job.result.return_value.to_dataframe.return_value = pd.DataFrame({
        "date": ["2026-02-01"],
        "spend": [100],
    })

    os.makedirs("data/clean", exist_ok=True)
    os.makedirs("data/forecast/arima_plus", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\n"
            "default:\n"
            "  date_column: date\n"
            "  prediction_column: spend\n"
            "cloud:\n"
            "  project_id: test_project\n"
            "  dataset_id: test_dataset\n"
            "data:\n"
            "  clean_folder: data/clean\n"
            "  forecast_folder: data/forecast\n"
        ),
    )
    self.fs.create_file(
        "data/clean/test_tag.csv",
        contents="date,spend\n2026-01-01,100\n2026-01-31,200\n",
    )

    forecast_arima_plus.execute_arima_plus_forecast("params.yaml")

    mock_client.query.assert_called_once()
    mock_upload.assert_called_once()
    self.assertTrue(
        os.path.exists("data/forecast/arima_plus/test_tag_forecast.csv")
    )

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_execute_arima_plus_forecast_missing_params(self, mock_upload):
    os.makedirs("data/clean_data", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\ndefault:\n  date_column: date\n  prediction_column:"
            " spend\ncloud:\n  project_id: null\n"
        ),
    )
    self.fs.create_file(
        "data/clean_data/test_tag.csv", contents="date,spend\n2026-01-01,100\n"
    )
    with self.assertRaises(ValueError):
      forecast_arima_plus.execute_arima_plus_forecast("params.yaml")
    mock_upload.assert_not_called()

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_execute_arima_plus_forecast_missing_source(self, mock_upload):
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\n"
            "cloud:\n"
            "  project_id: test_project\n"
            "  dataset_id: test_dataset\n"
            "data:\n"
            "  clean_folder: data/non_existent\n"
        ),
    )
    with self.assertLogs(level="ERROR") as log:
      forecast_arima_plus.execute_arima_plus_forecast("params.yaml")
      self.assertIn("Missing", log.records[0].getMessage())
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
