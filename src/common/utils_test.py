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

"""Tests for utils."""

from unittest import mock

from pyfakefs import fake_filesystem_unittest

from common import utils
from google3.testing.pybase import googletest


class UtilsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

    self.mock_storage_client_cls = self.enter_context(
        mock.patch("google.cloud.storage.Client", autospec=True)
    )
    self.mock_blob = mock.Mock()
    self.mock_bucket = mock.Mock()
    self.mock_bucket.blob.return_value = self.mock_blob
    self.mock_storage_client = self.mock_storage_client_cls.return_value
    self.mock_storage_client.bucket.return_value = self.mock_bucket

  def test_load_run_params_success(self):
    self.fs.create_file(
        "params.yaml",
        contents="tag: test_tag\ncloud:\n  bucket_name: my_bucket",
    )
    params = utils.load_run_params("params.yaml")
    self.assertEqual(params["tag"], "test_tag")
    self.assertEqual(params["cloud"]["bucket_name"], "my_bucket")

  def test_load_run_params_with_local_override(self):
    self.fs.create_file(
        "params.yaml",
        contents="tag: base_tag\ncloud:\n  bucket_name: base_bucket",
    )
    self.fs.create_file(
        "params_local.yaml",
        contents="tag: override_tag\ncloud:\n  project_id: new_project",
    )
    with self.assertLogs(level="INFO") as log:
      params = utils.load_run_params("params.yaml")
      self.assertTrue(any("override" in r.getMessage() for r in log.records))
    self.assertEqual(params["tag"], "override_tag")
    self.assertEqual(params["cloud"]["bucket_name"], "base_bucket")
    self.assertEqual(params["cloud"]["project_id"], "new_project")

  def test_load_run_params_fallback_to_base(self):
    self.fs.create_file(
        "params_base.yaml",
        contents="tag: fallback_tag\ncloud:\n  bucket_name: fallback_bucket",
    )
    with self.assertLogs(level="INFO") as log:
      params = utils.load_run_params("params.yaml")
      self.assertIn("baseline", log.records[0].getMessage())
    self.assertEqual(params["tag"], "fallback_tag")
    self.assertEqual(params["cloud"]["bucket_name"], "fallback_bucket")

  def test_load_run_params_file_not_found(self):
    with self.assertRaises(FileNotFoundError):
      utils.load_run_params("non_existent.yaml")

  def test_get_current_tag_success(self):
    params = {"tag": "my_tag"}
    self.assertEqual(utils.get_current_tag(params), "my_tag")

  def test_get_current_tag_missing(self):
    with self.assertRaises(KeyError):
      utils.get_current_tag({})

  def test_upload_to_gcs_success(self):
    self.fs.create_file("local_file.txt", contents="data")
    self.fs.create_file(
        "params.yaml", contents="cloud:\n  bucket_name: test_bucket"
    )

    utils.upload_to_gcs("local_file.txt")

    self.mock_storage_client.bucket.assert_called_once_with("test_bucket")
    self.mock_bucket.blob.assert_called_once_with("outputs/local_file.txt")
    self.mock_blob.upload_from_filename.assert_called_once_with(
        "local_file.txt"
    )

  def test_upload_to_gcs_with_bucket_override(self):
    self.fs.create_file("local_file.txt", contents="data")
    self.fs.create_file(
        "params.yaml", contents="cloud:\n  bucket_name: wrong_bucket"
    )

    utils.upload_to_gcs("local_file.txt", bucket_name="explicit_bucket")

    self.mock_storage_client.bucket.assert_called_once_with("explicit_bucket")
    self.mock_bucket.blob.assert_called_once_with("outputs/local_file.txt")
    self.mock_blob.upload_from_filename.assert_called_once_with(
        "local_file.txt"
    )

  def test_upload_to_gcs_no_local_file(self):
    with self.assertLogs(level="WARNING") as log:
      utils.upload_to_gcs("non_existent_local_file.txt")
      self.assertIn("missing", log.records[0].getMessage())
    self.mock_storage_client.bucket.assert_not_called()

  def test_upload_to_gcs_no_bucket(self):
    self.fs.create_file("local_file.txt", contents="data")
    self.fs.create_file("params.yaml", contents="cloud:\n  no_bucket: true")
    with self.assertLogs(level="ERROR") as log:
      utils.upload_to_gcs("local_file.txt")
      self.assertIn("No target bucket", log.records[0].getMessage())
    self.mock_storage_client.bucket.assert_not_called()

  def test_upload_to_gcs_upload_fails(self):
    self.fs.create_file("local_file.txt", contents="data")
    self.fs.create_file(
        "params.yaml", contents="cloud:\n  bucket_name: test_bucket"
    )
    self.mock_blob.upload_from_filename.side_effect = OSError(
        "Permission denied"
    )
    with self.assertLogs(level="ERROR") as log:
      utils.upload_to_gcs("local_file.txt")
      self.assertIn(
          "An unexpected error occurred during GCS upload",
          log.records[0].getMessage(),
      )


if __name__ == "__main__":
  googletest.main()
