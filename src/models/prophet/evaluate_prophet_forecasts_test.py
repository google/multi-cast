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

"""Tests for evaluate_prophet_forecasts."""

import os
from unittest import mock

import pandas as pd
from pyfakefs import fake_filesystem_unittest

from common import utils
from models.prophet import evaluate_prophet_forecasts
from google3.testing.pybase import googletest


class EvaluateProphetForecastsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_calculate_metrics_success(self):
    actuals = pd.Series([100.0, 200.0])
    forecasts = pd.Series([110.0, 190.0])
    metrics = evaluate_prophet_forecasts.calculate_metrics(actuals, forecasts)

    self.assertAlmostEqual(metrics["actual_sum"], 300.0)
    self.assertAlmostEqual(metrics["forecast_sum"], 300.0)
    self.assertAlmostEqual(metrics["deviation"], 0.0)
    self.assertIn("mape", metrics)

  def test_calculate_metrics_zero_actual_sum(self):
    actuals = pd.Series([0.0, 0.0])
    forecasts = pd.Series([10.0, 10.0])
    with self.assertRaises(ValueError):
      evaluate_prophet_forecasts.calculate_metrics(actuals, forecasts)

  def test_generate_evaluation_plot_success(self):
    plot_data = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
        "actual": [100.0, 200.0],
        "forecast": [110.0, 190.0],
        "lower_bound": [90.0, 180.0],
        "upper_bound": [120.0, 210.0],
    })
    p = evaluate_prophet_forecasts.generate_evaluation_plot(
        plot_data, pd.Timestamp("2026-01-15"), "TestTag", 0.05
    )
    self.assertIsNotNone(p)

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  @mock.patch.object(
      evaluate_prophet_forecasts.gg.ggplot, "save", autospec=True
  )
  def test_evaluate_prophet_precision_success(self, mock_save, mock_upload):
    os.makedirs("data/split", exist_ok=True)
    os.makedirs("data/forecast/prophet", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\n"
            "default:\n"
            "  date_column: date\n"
            "  prediction_column: spend\n"
            "data:\n"
            "  split_folder: data/split\n"
            "  forecast_folder: data/forecast\n"
            "  evaluate_folder: data/evaluate\n"
        ),
    )
    self.fs.create_file(
        "data/split/test_tag_train.csv",
        contents="date,spend\n2026-01-01,100\n2026-01-02,200\n",
    )
    self.fs.create_file(
        "data/split/test_tag_test.csv",
        contents="date,spend\n2026-01-03,300\n",
    )
    self.fs.create_file(
        "data/forecast/prophet/test_tag_forecast.csv",
        contents=(
            "ds,yhat,yhat_lower,yhat_upper\n"
            "2026-01-01,100,90,110\n"
            "2026-01-02,200,180,220\n"
            "2026-01-03,290,270,310\n"
        ),
    )

    evaluate_prophet_forecasts.evaluate_prophet_precision_and_plot(
        "params.yaml"
    )

    mock_save.assert_called_once()
    self.assertEqual(mock_upload.call_count, 2)
    self.assertTrue(
        os.path.exists("data/evaluate/prophet/test_tag_metrics.json")
    )

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_evaluate_prophet_precision_missing_prerequisites(self, mock_upload):
    self.fs.create_file(
        "params.yaml",
        contents="tag: test_tag\ndata:\n  split_folder: data/non_existent\n",
    )
    with self.assertLogs(level="ERROR") as log:
      evaluate_prophet_forecasts.evaluate_prophet_precision_and_plot(
          "params.yaml"
      )
      self.assertIn("Missing", log.records[0].getMessage())
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
