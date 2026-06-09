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

"""Tests for upload_to_bq."""

import io
import os
from unittest import mock

from google.cloud import bigquery
from pyfakefs import fake_filesystem_unittest

from common import utils
from pipelines import upload_to_bq
from google3.testing.pybase import googletest


class UploadToBqTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_calculate_file_md5_success(self):
    data = b"test_content"
    file_obj = io.BytesIO(data)
    checksum = upload_to_bq.calculate_file_md5(file_obj)
    self.assertEqual(checksum, "27565f9a57c128674736aa644012ce67")
    self.assertEqual(file_obj.tell(), 0)

  @mock.patch.object(bigquery, "Client", autospec=True)
  def test_execute_bigquery_load_success(self, mock_client_cls):
    mock_client = mock.Mock()
    mock_client_cls.return_value = mock_client
    mock_job = mock.Mock()
    mock_client.load_table_from_file.return_value = mock_job
    mock_table = mock.Mock()
    mock_table.num_rows = 42
    mock_client.get_table.return_value = mock_table

    file_obj = io.BytesIO(b"col1,col2\nval1,val2\n")
    num_rows = upload_to_bq.execute_bigquery_load(
        mock_client, file_obj, "project_id", "dataset_id", "table_id"
    )

    self.assertEqual(num_rows, 42)
    mock_client.create_dataset.assert_called_once()
    mock_client.load_table_from_file.assert_called_once()
    mock_job.result.assert_called_once()
    mock_client.get_table.assert_called_once()

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  @mock.patch.object(upload_to_bq, "execute_bigquery_load", autospec=True)
  @mock.patch.object(bigquery, "Client", autospec=True)
  def test_upload_target_dataset_success(
      self, mock_execute_load, mock_upload
  ):
    mock_execute_load.return_value = 10
    os.makedirs("data/clean", exist_ok=True)
    self.fs.create_file(
        "params.yaml",
        contents=(
            "tag: test_tag\n"
            "cloud:\n"
            "  project_id: test_project\n"
            "  dataset_id: test_dataset\n"
            "data:\n"
            "  clean_folder: data/clean\n"
            "  upload_folder: data/upload\n"
        ),
    )
    self.fs.create_file("data/clean/test_tag.csv", contents="col1\nval1\n")

    upload_to_bq.upload_target_dataset_to_bq("params.yaml")

    mock_execute_load.assert_called_once()
    mock_upload.assert_called_once()
    self.assertTrue(os.path.exists("data/upload/test_tag_hash.txt"))

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_upload_target_dataset_missing_params(self, mock_upload):
    self.fs.create_file(
        "params.yaml", contents="tag: test_tag\ncloud:\n  project_id: null\n"
    )
    with self.assertRaises(ValueError):
      upload_to_bq.upload_target_dataset_to_bq("params.yaml")
    mock_upload.assert_not_called()

  @mock.patch.object(utils, "upload_to_gcs", autospec=True)
  def test_upload_target_dataset_missing_source(self, mock_upload):
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
      upload_to_bq.upload_target_dataset_to_bq("params.yaml")
      self.assertIn("Missing", log.records[0].getMessage())
    mock_upload.assert_not_called()


if __name__ == "__main__":
  googletest.main()
