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

"""Primary execution entrypoint for Multi-Cast batch forecasting under Cloud Run Jobs.

Discovers overriding payload parameters passed securely via environment
variables.
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from absl import logging
import yaml

from common import utils


def clear_dangling_locks() -> None:
  """Removes leftover DVC read/write transaction locks preventing execution bottlenecks."""
  lock_file = ".dvc/tmp/rwlock"
  if os.path.exists(lock_file):
    try:
      os.remove(lock_file)
      logging.info("Cleared dangling DVC lock successfully.")
    except OSError as e:
      logging.warning("Failed to clear dangling DVC lock: %s", e)


def build_merged_params(
    sys_argv: List[str],
    environ: Dict[str, str],
    config_template_path: str = "params_base.yaml",
) -> Dict[str, Any]:
  """Extracts overriding payload configurations and merges them with base parameters.

  Args:
      sys_argv: Command line arguments list.
      environ: Environment variables dictionary.
      config_template_path: Path to the base configuration YAML template.

  Returns:
      Dictionary containing fully merged configuration parameters.

  Raises:
      ValueError: If provided JSON payloads are invalid.
  """
  if len(sys_argv) > 1:
    raw_payload = " ".join(sys_argv[1:])
  else:
    raw_payload = environ.get("JOB_PAYLOAD_JSON", "{}")

  try:
    payload_dict = json.loads(raw_payload)
  except json.JSONDecodeError as e:
    logging.error("Invalid JSON format provided in payload: %s", raw_payload)
    raise ValueError(f"Invalid JSON payload: {e}") from e

  # Load base parameters safely
  current_params: Dict[str, Any] = {}
  if os.path.exists(config_template_path):
    with open(config_template_path, "r") as f:
      current_params = yaml.safe_load(f) or {}
  else:
    logging.warning(
        "Base configuration template %s not found. Starting with empty params.",
        config_template_path,
    )

  # Merge global payload configuration
  utils._dict_merge(current_params, payload_dict)

  # Retrieve and merge task-specific parameters based on Cloud Run Task Index
  task_params_str = environ.get("taskParameters") or environ.get(
      "TASK_PARAMETERS"
  )
  if task_params_str:
    try:
      tasks_list = json.loads(task_params_str)
      if isinstance(tasks_list, list):
        task_index = int(environ.get("CLOUD_RUN_TASK_INDEX", 0))
        if 0 <= task_index < len(tasks_list):
          utils._dict_merge(current_params, tasks_list[task_index])
          logging.info(
              "Merged task-specific parameters for task index: %d", task_index
          )
        else:
          logging.warning(
              "Task index %d out of bounds for taskParameters array.",
              task_index,
          )
    except json.JSONDecodeError as e:
      logging.error(
          "Invalid JSON format in taskParameters: %s", task_params_str
      )
      raise ValueError(f"Invalid taskParameters JSON: {e}") from e

  return current_params


def execute_batch_pipeline(
    sys_argv: Optional[List[str]] = None,
    environ: Optional[Dict[str, str]] = None,
    config_template_path: str = "params_base.yaml",
    config_path: str = "params.yaml",
) -> None:
  """Coordinates configuration injection and synchronous DVC execution pipeline.

  Args:
      sys_argv: Optional command-line arguments override.
      environ: Optional environment variables override.
      config_template_path: Source base configuration template path.
      config_path: Destination configuration file path.
  """
  if sys_argv is None:
    sys_argv = sys.argv
  if environ is None:
    environ = dict(os.environ)

  try:
    merged_params = build_merged_params(sys_argv, environ, config_template_path)
  except ValueError:
    sys.exit(1)

  with open(config_path, "w") as f:
    yaml.dump(merged_params, f)

  logging.info(
      "Configuration payload successfully injected. Initiating pipeline"
      " reproduction."
  )

  clear_dangling_locks()

  try:
    subprocess.run(
        ["dvc", "repro"],
        check=True,
    )
    logging.info(
        "DVC Forecasting pipeline successfully concluded in batch mode."
    )
  except subprocess.CalledProcessError as e:
    logging.error("Critical DVC pipeline execution failure: %s", e)
    sys.exit(1)


if __name__ == "__main__":
  execute_batch_pipeline()
