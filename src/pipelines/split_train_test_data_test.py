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

"""Tests for split_train_test_data."""

import os
from unittest import mock

import pandas as pd
from pyfakefs import fake_filesystem_unittest

from pipelines import split_train_test_data
from common import utils
from google3.testing.pybase import googletest


class SplitTrainTestDataTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_split_dataframe_success(self):
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2026-01-01",
            "2026-01-15",
            "2026-01-20",
            "2026-01-25",
            "2026-01-30",
        ]),
        "val": [1, 2, 3, 4, 5],
    })
    train_df, test_df = split_train_test_data.split_dataframe(
        df, split_days=10, date_col="date"
    )
    self.assertEqual(len(train_df), 2)
    self.assertEqual(len(test_df), 3)
    self.assertEqual(train_df["date"].max(), pd.Timestamp("2026-01-15"))
    self.assertEqual(test_df["date"].min(), pd.Timestamp("2026-01-20"))

  def test_split_dataframe_empty(self):
    df = pd.DataFrame(columns=["date", "val"])
    with self.assertRaises(ValueError):
      split_train_test_data.split_dataframe(df, split_days=10, date_col="date")

  def test_split_dataframe_missing_date(self):
    df = pd.DataFrame({"val": [1, 2]})
    with self.assertRaises(ValueError):
      split_train_test_data.split_dataframe(df, split_days=10, date_col="date")

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_split_temporal_data_success(self, mock_upload):
    os.makedirs("data/clean", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\n"
            "default:\n"
            "  split_train_and_test: 5\n"
            "  date_column: date\n"
            "data:\n"
            "  clean_folder: data/clean\n"
            "  split_folder: data/split\n"
        ),
    )
    self.fs.create_file(
        "data/clean/test_tag.csv",
        contents="date,val\n2026-01-01,10\n2026-01-10,20\n2026-01-12,30\n",
    )

    split_train_test_data.split_temporal_data("params.yaml")

    train_csv = os.path.join("data", "split", "test_tag_train.csv")
    test_csv = os.path.join("data", "split", "test_tag_test.csv")
    self.assertTrue(os.path.exists(train_csv))
    self.assertTrue(os.path.exists(test_csv))
    self.assertEqual(mock_upload.call_count, 2)

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_split_temporal_data_missing_source(self, mock_upload):
    self.fs.create_file(
        "params.yaml",
        contents="tag: test_tag\ndata:\n  clean_folder: data/non_existent\n",
    )
    with self.assertLogs(level="ERROR") as log:
      split_train_test_data.split_temporal_data("params.yaml")
      self.assertIn("missing", log.records[0].getMessage())
    mock_upload.assert_not_called()

  def test_split_temporal_data_invalid_split_param(self):
    self.fs.create_file(
        "params.yaml",
        contents="tag: test_tag\ndefault:\n  split_train_and_test: -10\n",
    )
    with self.assertRaises(ValueError):
      split_train_test_data.split_temporal_data("params.yaml")


if __name__ == "__main__":
  googletest.main()
