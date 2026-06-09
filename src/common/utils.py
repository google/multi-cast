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

"""Shared configuration and environment utility module for forecasting pipelines.

Abstracts loading parameters and strictly validating context tags.
"""

import os
from typing import Any, Dict, Optional

from absl import logging
from google.api_core import exceptions as api_core_exceptions
from google.auth import exceptions as auth_exceptions
from google.cloud import storage
import yaml


def _dict_merge(dct: Dict[str, Any], merge_dct: Dict[str, Any]) -> None:
  """Recursively merges merge_dct into dct.

  Existing dictionaries are updated recursively. Other values in dct are
  overwritten by values from merge_dct.

  Args:
      dct: Target dictionary to be updated in place.
      merge_dct: Source dictionary containing override values.
  """
  for k, v in merge_dct.items():
    if k in dct and isinstance(dct[k], dict) and isinstance(v, dict):
      _dict_merge(dct[k], v)
    else:
      dct[k] = v


def load_run_params(config_path: str = "params.yaml") -> Dict[str, Any]:
  """Safely loads pipeline configuration attributes with smart fallback support.

  Attempts to load configuration from config_path (e.g., DVC generated
  params.yaml). If config_path does not exist (e.g., standalone script debugging
  mode), it intelligently falls back to reading the immutable baseline
  configuration params_base.yaml. In both modes, if a companion local file
  named `<base>_local.yaml` (e.g., params_local.yaml) exists, it is loaded and
  recursively merged into the configuration.

  Args:
      config_path: Target parameter descriptor path.

  Returns:
      Dictionary representing merged parameter namespaces.

  Raises:
      FileNotFoundError: if neither config_path nor base configuration exists.
  """
  params = {}
  if os.path.exists(config_path):
    logging.info("Loading primary configuration file: %s", config_path)
    with open(config_path, "r") as f:
      params = yaml.safe_load(f) or {}
  else:
    base_config_path = config_path.replace("params.yaml", "params_base.yaml")
    if os.path.exists(base_config_path):
      logging.info(
          "Primary configuration not found; falling back to baseline: %s",
          base_config_path,
      )
      with open(base_config_path, "r") as bf:
        params = yaml.safe_load(bf) or {}
    else:
      logging.error(
          "Fatal error: Neither primary %s nor baseline %s found.",
          config_path,
          base_config_path,
      )
      raise FileNotFoundError(
          f"No configuration found at {config_path} or {base_config_path}"
      )

  # Check for local uncommitted configuration override file
  local_config_path = config_path.replace(".yaml", "_local.yaml")
  if os.path.exists(local_config_path):
    logging.info(
        "Loading local private configuration override from %s",
        local_config_path,
    )
    with open(local_config_path, "r") as lf:
      local_params = yaml.safe_load(lf) or {}
      _dict_merge(params, local_params)

  return params


def get_current_tag(params: Dict[str, Any]) -> str:
  """Strictly extracts the running target tag from configured attributes.

  Args:
      params: Parsed parameter options.

  Raises:
      KeyError: When the 'current' configuration or 'tag' attribute is absent.

  Returns:
      String descriptor for the execution context.
  """
  tag = params.get("tag")
  if not tag:
    raise KeyError(
        "Mandatory configuration parameter 'tag' is missing or empty."
    )
  return tag


def get_date_col(params: Dict[str, Any]) -> str:
  """Retrieves the configured date column name."""
  return params.get("default", {}).get("date_column", "date_id")


def get_val_col(params: Dict[str, Any]) -> str:
  """Retrieves the configured prediction value column name."""
  return params.get("default", {}).get("prediction_column", "spend")


def upload_to_gcs(
    local_filepath: str,
    bucket_name: Optional[str] = None,
    gcs_blob_path: Optional[str] = None,
) -> None:
  """Uploads artifact files deriving destination defaults from local configuration parameters.

  Args:
      local_filepath: Source artifact file descriptor path.
      bucket_name: Target Google Cloud Storage bucket identifier.
      gcs_blob_path: Path within bucket.
  """
  if not os.path.exists(local_filepath):
    logging.warning(
        "Warning: Local file %s missing. Cannot upload.",
        local_filepath,
    )
    return

  try:
    params = load_run_params()
    cloud_config = params.get("cloud", {})

    if bucket_name is None:
      bucket_name = cloud_config.get("bucket_name")
    if not bucket_name:
      logging.error(
          "Error: No target bucket specified or found in configuration."
      )
      return

    if gcs_blob_path is None:
      upload_path = cloud_config.get("upload_path", "outputs")
      gcs_blob_path = f"{upload_path}/{local_filepath.lstrip('/')}"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)

    blob.upload_from_filename(local_filepath)
    logging.info(
        "Archived artifact %s to gs://%s/%s",
        local_filepath,
        bucket_name,
        gcs_blob_path,
    )

  except (
      api_core_exceptions.GoogleAPICallError,
      api_core_exceptions.RetryError,
      auth_exceptions.GoogleAuthError,
      yaml.YAMLError,
      OSError,
  ) as e:
    logging.error(
        "An unexpected error occurred during GCS upload of %s: %s",
        local_filepath,
        e,
    )
