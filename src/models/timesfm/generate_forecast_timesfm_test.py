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

"""Tests for generate_forecast_timesfm."""

import os
from unittest import mock

from google.cloud import bigquery
import pandas as pd
from pyfakefs import fake_filesystem_unittest

from common import utils
from models.timesfm import generate_forecast_timesfm
from google3.testing.pybase import googletest


class GenerateForecastTimesfmTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_build_generate_timesfm_query_success(self):
    query = generate_forecast_timesfm.build_generate_timesfm_query(
        "proj", "dataset", "table", "2026-04-01", 30, 0.95, "date", "spend"
    )
    self.assertIn("proj.dataset.table", query)
    self.assertIn("2026-04-01", query)
    self.assertIn("horizon => 30", query)
    self.assertIn("confidence_level => 0.95", query)

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  @mock.patch.object(bigquery, "Client", autospec=True)
  def test_generate_final_timesfm_sequence_success(
      self, mock_client_cls, mock_upload
  ):
    mock_client = mock.Mock()
    mock_client_cls.return_value = mock_client
    mock_job = mock.Mock()
    mock_client.query.return_value = mock_job
    mock_job.result.return_value.to_dataframe.return_value = pd.DataFrame({
        "forecast_timestamp": ["2026-04-01", "2026-04-02"],
        "spend": [100, 200],
    })

    os.makedirs("data/split", exist_ok=True)
    os.makedirs("data/generate_forecast/timesfm", exist_ok=True)
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
            "  split_folder: data/split\n"
            "  generate_forecast_folder: data/generate_forecast\n"
            "generate_forecast:\n"
            "  start_date: '2026-04-01'\n"
            "  end_date: '2026-04-02'\n"
        ),
    )
    self.fs.create_file(
        "data/split/test_tag_train.csv",
        contents="date,spend\n2026-01-01,100\n2026-03-31,200\n",
    )

    generate_forecast_timesfm.generate_final_timesfm_sequence("params.yaml")

    mock_client.query.assert_called_once()
    mock_upload.assert_called_once()
    self.assertTrue(
        os.path.exists("data/generate_forecast/timesfm/test_tag_forecast.csv")
    )

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_generate_final_timesfm_missing_params(self, mock_upload):
    os.makedirs("data/split_data", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\ndefault:\n  date_column: date\n  prediction_column:"
            " spend\ncloud:\n  project_id: null\n"
        ),
    )
    self.fs.create_file(
        "data/split_data/test_tag_train.csv",
        contents="date,spend\n2026-01-01,100\n",
    )
    with self.assertRaises(ValueError):
      generate_forecast_timesfm.generate_final_timesfm_sequence("params.yaml")
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
