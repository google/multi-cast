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

"""Tests for generate_forecast_prophet."""

import os
from unittest import mock

import numpy as np
import pandas as pd
from pyfakefs import fake_filesystem_unittest

from common import utils
from models.prophet import generate_forecast_prophet
from google3.testing.pybase import googletest


class GenerateForecastProphetTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_prepare_forecasting_data_success(self):
    self.fs.create_file(
        "clean.csv", contents="date,spend\n2026-01-01,100\n2026-01-02,200\n"
    )
    df = generate_forecast_prophet.prepare_forecasting_data(
        "clean.csv", "date", "spend"
    )
    self.assertLen(df, 2)
    self.assertIn("ds", df.columns)
    self.assertIn("y", df.columns)

  def test_prepare_forecasting_data_missing_file(self):
    df = generate_forecast_prophet.prepare_forecasting_data(
        "non_existent.csv", "date", "spend"
    )
    self.assertTrue(df.empty)

  def test_prepare_forecasting_data_missing_columns(self):
    self.fs.create_file("invalid.csv", contents="date,other\n2026-01-01,100\n")
    with self.assertRaises(ValueError):
      generate_forecast_prophet.prepare_forecasting_data(
          "invalid.csv", "date", "spend"
      )

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  @mock.patch.object(generate_forecast_prophet, "Prophet", autospec=True)
  @mock.patch.object(
      generate_forecast_prophet, "model_from_json", autospec=True
  )
  def test_generate_final_sequence_success(
      self, mock_from_json, mock_prophet_cls, mock_upload
  ):
    mock_model_stub = mock.Mock()
    mock_model_stub.changepoint_prior_scale = 0.05
    mock_model_stub.changepoint_range = 0.8
    mock_model_stub.seasonality_mode = "multiplicative"
    mock_model_stub.seasonality_prior_scale = 10.0
    mock_model_stub.holidays_prior_scale = 10.0
    mock_from_json.return_value = mock_model_stub

    mock_model = mock.Mock()
    mock_prophet_cls.return_value = mock_model
    mock_model.predict.return_value = pd.DataFrame({
        "ds": ["2026-04-01", "2026-04-02"],
        "yhat": [np.log1p(100), np.log1p(200)],
        "yhat_lower": [np.log1p(90), np.log1p(190)],
        "yhat_upper": [np.log1p(110), np.log1p(210)],
    })

    os.makedirs("data/clean_data", exist_ok=True)
    os.makedirs("data/forecast/prophet", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\n"
            "default:\n"
            "  date_column: date\n"
            "  prediction_column: spend\n"
            "data:\n"
            "  clean_folder: data/clean_data\n"
            "  forecast_folder: data/forecast\n"
            "  generate_forecast_folder: data/generate_forecast\n"
            "generate_forecast:\n"
            "  start_date: '2026-04-01'\n"
            "  end_date: '2026-04-02'\n"
        ),
    )
    self.fs.create_file(
        "data/clean_data/test_tag.csv",
        contents="date,spend\n2026-01-01,100\n2026-01-02,200\n",
    )
    self.fs.create_file(
        "data/forecast/prophet/test_tag_model.json",
        contents='{"model": "stub"}',
    )

    generate_forecast_prophet.generate_final_sequence("params.yaml")

    mock_model.fit.assert_called_once()
    mock_model.predict.assert_called_once()
    mock_upload.assert_called_once()
    self.assertTrue(
        os.path.exists("data/generate_forecast/prophet/test_tag_forecast.csv")
    )

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_generate_final_sequence_missing_prerequisites(self, mock_upload):
    self.fs.create_file(
        "params.yaml",
        contents="tag: test_tag\ndata:\n  clean_folder: data/non_existent\n",
    )
    with self.assertLogs(level="ERROR") as log:
      generate_forecast_prophet.generate_final_sequence("params.yaml")
      self.assertIn("Missing", log.records[0].getMessage())
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
