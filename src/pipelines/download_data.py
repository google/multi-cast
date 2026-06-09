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

"""Retrieves incoming raw datasets from Google Cloud Storage using Python SDK.

Eliminates the need for heavyweight gcloud CLI binary dependencies within
containerized environments.
"""

import os
import sys
from typing import Optional

from absl import logging
from google.cloud import storage

from common import utils


def download_incoming_data(config_path: str = "params.yaml") -> None:
  """Downloads configured incoming raw data file from GCS to local staging path.

  Args:
      config_path: Path to the YAML configuration file.

  Raises:
      ValueError: If mandatory cloud storage parameters are missing.
      RuntimeError: If GCS download or local file staging fails.
  """
  params = utils.load_run_params(config_path)

  cloud_config = params.get("cloud", {})
  bucket_name: Optional[str] = cloud_config.get("bucket_name")
  incoming_file: Optional[str] = cloud_config.get("incoming_file")

  data_config = params.get("data", {})
  local_file: str = data_config.get("local_file", "data/raw/source.csv")

  if not bucket_name or not incoming_file:
    logging.error(
        "Mandatory GCS parameters 'bucket_name' or 'incoming_file' missing"
        " from configuration."
    )
    raise ValueError(
        "Missing essential GCS parameters in cloud configuration namespace."
    )

  # Ensure local destination directory structure exists securely
  try:
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
  except OSError as e:
    logging.exception(
        "Failed to create local staging directory for %s: %s", local_file, e
    )
    raise RuntimeError(f"Local directory staging error: {e}") from e

  try:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(incoming_file)

    logging.info(
        "Downloading raw dataset from gs://%s/%s to %s",
        bucket_name,
        incoming_file,
        local_file,
    )
    blob.download_to_filename(local_file)
    logging.info(
        "Successfully retrieved incoming dataset and staged at: %s", local_file
    )

  except (Exception, OSError) as e:
    logging.exception(
        "Critical failure occurred while downloading gs://%s/%s: %s",
        bucket_name,
        incoming_file,
        e,
    )
    raise RuntimeError(f"GCS transfer failure: {e}") from e


def main() -> None:
  """Main execution wrapper handling top-level system exit upon failure."""
  try:
    download_incoming_data()
  except (ValueError, RuntimeError):
    sys.exit(1)


if __name__ == "__main__":
  main()
