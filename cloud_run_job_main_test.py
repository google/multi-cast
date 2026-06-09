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

"""Tests for cloud_run_job_main."""

import os
import subprocess
from unittest import mock

from pyfakefs import fake_filesystem_unittest
import yaml

from google3.corp.gtech.ads.solutions.multi_cast import cloud_run_job_main
from google3.testing.pybase import googletest


class CloudRunJobMainTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = fake_filesystem_unittest.Patcher()
    self.patcher.setUp()
    self.fs = self.patcher.fs
    self.addCleanup(self.patcher.tearDown)

  def test_clear_dangling_locks_success(self):
    os.makedirs(".dvc/tmp", exist_ok=True)
    lock_file = ".dvc/tmp/rwlock"
    self.fs.create_file(lock_file, contents="lock")
    self.assertTrue(os.path.exists(lock_file))

    cloud_run_job_main.clear_dangling_locks()
    self.assertFalse(os.path.exists(lock_file))

  def test_build_merged_params_sys_argv_override(self):
    self.fs.create_file(
        "params_base.yaml", contents="base_key: base_val\noverride_key: old_val"
    )
    sys_argv = ["main.py", '{"override_key": "new_val", "added_key": 123}']
    environ = {}

    merged = cloud_run_job_main.build_merged_params(
        sys_argv, environ, "params_base.yaml"
    )
    self.assertEqual(merged["base_key"], "base_val")
    self.assertEqual(merged["override_key"], "new_val")
    self.assertEqual(merged["added_key"], 123)

  def test_build_merged_params_environ_override_and_tasks(self):
    self.fs.create_file("params_base.yaml", contents="tag: base_tag")
    sys_argv = ["main.py"]
    environ = {
        "JOB_PAYLOAD_JSON": '{"global_setting": true}',
        "TASK_PARAMETERS": '[{"tag": "task_0_tag"}, {"tag": "task_1_tag"}]',
        "CLOUD_RUN_TASK_INDEX": "1",
    }

    merged = cloud_run_job_main.build_merged_params(
        sys_argv, environ, "params_base.yaml"
    )
    self.assertEqual(merged["tag"], "task_1_tag")
    self.assertTrue(merged["global_setting"])

  def test_build_merged_params_invalid_json(self):
    sys_argv = ["main.py", "invalid_json"]
    environ = {}
    with self.assertRaises(ValueError):
      cloud_run_job_main.build_merged_params(
          sys_argv, environ, "params_base.yaml"
      )

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_execute_batch_pipeline_success(self, mock_run):
    self.fs.create_file("params_base.yaml", contents="tag: test_tag")
    sys_argv = ["main.py", '{"param": "value"}']
    environ = {}

    cloud_run_job_main.execute_batch_pipeline(
        sys_argv, environ, "params_base.yaml", "params.yaml"
    )

    mock_run.assert_called_once_with(["dvc", "repro"], check=True)
    self.assertTrue(os.path.exists("params.yaml"))
    with open("params.yaml", "r") as f:
      params = yaml.safe_load(f)
      self.assertEqual(params["tag"], "test_tag")
      self.assertEqual(params["param"], "value")

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_execute_batch_pipeline_subprocess_failure(self, mock_run):
    self.fs.create_file("params_base.yaml", contents="tag: test")
    mock_run.side_effect = subprocess.CalledProcessError(1, ["dvc", "repro"])
    sys_argv = ["main.py"]
    environ = {}

    with self.assertRaises(SystemExit):
      cloud_run_job_main.execute_batch_pipeline(
          sys_argv, environ, "params_base.yaml", "params.yaml"
      )


if __name__ == "__main__":
  googletest.main()
