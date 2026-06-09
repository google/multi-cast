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

"""Tests for forecast_prophet."""

import os
from unittest import mock

import numpy as np
import optuna
import pandas as pd
from pyfakefs import fake_filesystem_unittest

from common import utils
from models.prophet import forecast_prophet
from google3.testing.pybase import googletest


class ForecastProphetTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_validate_dataframe_success(self):
    df = pd.DataFrame({"date": ["2026-01-01"], "spend": [100.0]})
    forecast_prophet._validate_dataframe(df, ["date", "spend"])

  def test_validate_dataframe_missing_columns(self):
    df = pd.DataFrame({"date": ["2026-01-01"]})
    with self.assertRaises(ValueError):
      forecast_prophet._validate_dataframe(df, ["date", "spend"])

  @mock.patch.object(forecast_prophet, "cross_validation", autospec=True)
  @mock.patch.object(forecast_prophet, "Prophet", autospec=True)
  def test_objective_success(self, mock_prophet_cls, mock_cv):
    mock_model = mock.Mock()
    mock_prophet_cls.return_value = mock_model
    mock_cv.return_value = pd.DataFrame({
        "y": [np.log1p(100), np.log1p(200)],
        "yhat": [np.log1p(100), np.log1p(200)],
    })

    # Mock optuna trial
    mock_trial = mock.create_autospec(optuna.Trial, instance=True)
    mock_trial.suggest_float.side_effect = [0.1, 0.5, 0.1, 0.1]
    mock_trial.suggest_categorical.side_effect = ["multiplicative", True, True]

    df = pd.DataFrame({"ds": ["2026-01-01"], "y": [100]})
    config = {
        "cp_prior_min": 0.05,
        "cp_prior_max": 5.0,
        "cp_range_min": 0.1,
        "cp_range_max": 0.9,
        "ssn_mode": ["multiplicative"],
        "ssn_prior_min": 0.05,
        "ssn_prior_max": 5.0,
        "holidays_prior_min": 0.05,
        "holidays_prior_max": 5.0,
        "country_name": "AU",
        "initial": "10 days",
        "period": "5 days",
        "horizon": "5 days",
        "cv_parallel": "None",
    }

    loss = forecast_prophet.objective(mock_trial, df, config)

    self.assertEqual(loss, 0.0)  # abs(300 - 300)
    mock_model.fit.assert_called_once_with(df)
    mock_cv.assert_called_once_with(
        model=mock_model,
        initial="10 days",
        period="5 days",
        horizon="5 days",
        parallel=None,
    )

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  @mock.patch.object(optuna, "create_study", autospec=True)
  @mock.patch.object(forecast_prophet, "Prophet", autospec=True)
  @mock.patch.object(forecast_prophet, "model_to_json", autospec=True)
  def test_run_prophet_training_success(
      self, mock_to_json, mock_prophet_cls, mock_create_study, mock_upload
  ):
    mock_study = mock.Mock()
    mock_study.best_params = {
        "changepoint_prior_scale": 0.1,
        "changepoint_range": 0.5,
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "seasonality_prior_scale": 0.1,
        "holidays_prior_scale": 0.1,
    }
    mock_create_study.return_value = mock_study

    mock_model = mock.Mock()
    mock_prophet_cls.return_value = mock_model
    mock_model.predict.return_value = pd.DataFrame({
        "ds": ["2026-01-01"],
        "yhat": [np.log1p(100)],
        "yhat_lower": [np.log1p(90)],
        "yhat_upper": [np.log1p(110)],
    })
    mock_fig = mock.Mock()
    mock_model.plot.return_value = mock_fig
    mock_to_json.return_value = '{"model": "data"}'

    os.makedirs("data/split", exist_ok=True)
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

    forecast_prophet.run_prophet_training_with_optimization("params.yaml")

    mock_study.optimize.assert_called_once()
    mock_model.fit.assert_called_once()
    mock_model.predict.assert_called_once()
    mock_fig.savefig.assert_called_once()
    self.assertEqual(mock_upload.call_count, 3)

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_run_prophet_training_missing_train(self, mock_upload):
    self.fs.create_file(
        "params.yaml",
        contents="tag: test_tag\ndata:\n  split_folder: data/non_existent\n",
    )
    with self.assertLogs(level="ERROR") as log:
      forecast_prophet.run_prophet_training_with_optimization("params.yaml")
      self.assertIn("missing", log.records[0].getMessage())
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
