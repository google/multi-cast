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

"""Tests for clean_data."""

import os
from unittest import mock

import pandas as pd
from pyfakefs import fake_filesystem_unittest

from pipelines import clean_data
from common import utils
from google3.testing.pybase import googletest


class CleanDataTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_preprocess_dataframe_success(self):
    df_raw = pd.DataFrame({
        "date_id": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "brand": ["BrandA", "BrandB", "BrandA"],
        "sales": [100, 200, 150],
    })
    # Filter for BrandA
    df_filtered = clean_data.preprocess_dataframe(
        df_raw, "brand == 'BrandA'", "date_id"
    )

    self.assertEqual(df_filtered.index.name, "date_id")
    self.assertEqual(len(df_filtered), 2)
    self.assertListEqual(df_filtered["sales"].tolist(), [100, 150])

  def test_preprocess_dataframe_missing_date_id(self):
    df_invalid = pd.DataFrame({"brand": ["BrandA"], "sales": [100]})
    with self.assertRaises(ValueError):
      clean_data.preprocess_dataframe(
          df_invalid, "brand == 'BrandA'", "date_id"
      )

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_clean_and_export_data_success(self, mock_upload):
    # Setup mock files
    os.makedirs("data/raw", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_run\n"
            "data_filter: \"brand == 'BrandA'\"\n"
            "default:\n"
            "  date_column: date_id\n"
            "  prediction_column: sales\n"
            "data:\n"
            "  local_file: data/raw/input.csv\n"
            "  clean_folder: data/clean\n"
        ),
    )
    self.fs.create_file(
        "data/raw/input.csv",
        contents="date_id,brand,sales\n2026-01-01,BrandA,100\n2026-01-02,BrandB,200\n",
    )

    clean_data.clean_and_export_data("params.yaml")

    expected_clean_csv = os.path.join("data", "clean", "test_run.csv")
    self.assertTrue(os.path.exists(expected_clean_csv))
    mock_upload.assert_called_once_with(local_filepath=expected_clean_csv)

    # Verify output CSV content
    df_out = pd.read_csv(expected_clean_csv)
    self.assertEqual(len(df_out), 1)
    self.assertEqual(df_out["brand"].iloc[0], "BrandA")

  def test_clean_and_export_data_missing_filter(self):
    self.fs.create_file(
        "params.yaml",
        contents="tag: test_run\ndata:\n  local_file: data/raw/input.csv\n",
    )
    with self.assertRaises(KeyError):
      clean_data.clean_and_export_data("params.yaml")

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_clean_and_export_data_source_missing(self, mock_upload):
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_run\n"
            "data_filter: \"brand == 'BrandA'\"\n"
            "data:\n"
            "  local_file: data/raw/non_existent.csv\n"
        ),
    )
    with self.assertLogs(level="ERROR") as log:
      clean_data.clean_and_export_data("params.yaml")
      self.assertIn("missing", log.records[0].getMessage())
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
