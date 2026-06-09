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

"""Tests for download_data."""

import os
from unittest import mock

from google.cloud import storage
from pyfakefs import fake_filesystem_unittest

from pipelines import download_data
from google3.testing.pybase import googletest


class DownloadDataTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  @mock.patch.object(storage, "Client", autospec=True)
  def test_download_incoming_data_success(self, mock_client_cls):
    mock_client = mock.Mock()
    mock_client_cls.return_value = mock_client
    mock_bucket = mock.Mock()
    mock_client.bucket.return_value = mock_bucket
    mock_blob = mock.Mock()
    mock_bucket.blob.return_value = mock_blob

    # Simulate successful download by creating the file in the mock blob call
    def side_effect_download(filename):
      self.fs.create_file(filename, contents="date,spend\n2026-01-01,100\n")

    mock_blob.download_to_filename.side_effect = side_effect_download

    self.fs.create_file(
        "params.yaml",
        contents=(
            "cloud:\n"
            "  bucket_name: test_bucket\n"
            "  incoming_file: test_file.csv\n"
            "data:\n"
            "  local_file: data/raw/test.csv\n"
        ),
    )

    download_data.download_incoming_data("params.yaml")

    mock_client_cls.assert_called_once()
    mock_client.bucket.assert_called_once_with("test_bucket")
    mock_bucket.blob.assert_called_once_with("test_file.csv")
    mock_blob.download_to_filename.assert_called_once_with("data/raw/test.csv")
    self.assertTrue(os.path.exists("data/raw/test.csv"))

  def test_download_incoming_data_missing_params(self):
    self.fs.create_file("params.yaml", contents="cloud:\n  bucket_name: null\n")
    with self.assertRaises(ValueError):
      download_data.download_incoming_data("params.yaml")

  @mock.patch.object(storage, "Client", autospec=True)
  def test_download_incoming_data_gcs_failure(self, mock_client_cls):
    mock_client = mock.Mock()
    mock_client_cls.return_value = mock_client
    mock_client.bucket.side_effect = Exception("GCS API Error")

    self.fs.create_file(
        "params.yaml",
        contents=(
            "cloud:\n  bucket_name: test_bucket\n  incoming_file: test.csv\n"
        ),
    )

    with self.assertRaises(RuntimeError):
      download_data.download_incoming_data("params.yaml")


if __name__ == "__main__":
  googletest.main()
