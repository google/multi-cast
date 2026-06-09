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

"""Transfers target cleaned forecasting dataset segments to BigQuery tables.

Applies shared configuration helpers and isolates executing context matching
project conventions.
"""

import hashlib
import os
from typing import BinaryIO

from absl import logging
from google.cloud import bigquery

from common import utils


def calculate_file_md5(file_obj: BinaryIO) -> str:
  """Computes the MD5 hexadecimal checksum of a binary file stream.

  Args:
      file_obj: Open binary file stream.

  Returns:
      MD5 checksum string.
  """
  file_obj.seek(0)
  checksum = hashlib.md5(file_obj.read()).hexdigest()
  file_obj.seek(0)
  return checksum


def execute_bigquery_load(
    client: bigquery.Client,
    file_obj: BinaryIO,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> int:
  """Configures and executes a BigQuery CSV load job.

  Args:
      client: Initialized BigQuery client.
      file_obj: Binary stream of the CSV file to upload.
      project_id: Target GCP project identifier.
      dataset_id: Target BigQuery dataset identifier.
      table_id: Destination table name.

  Returns:
      Total number of rows loaded into the destination table.
  """
  dataset_ref = f"{project_id}.{dataset_id}"
  logging.info("Verifying or creating BigQuery dataset: %s", dataset_ref)
  client.create_dataset(dataset_ref, exists_ok=True)

  job_config = bigquery.LoadJobConfig(
      source_format=bigquery.SourceFormat.CSV,
      skip_leading_rows=1,
      autodetect=True,
      write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
  )

  table_ref = f"{dataset_ref}.{table_id}"
  logging.info("Initiating BigQuery load job against table: %s", table_ref)

  file_obj.seek(0)
  job = client.load_table_from_file(file_obj, table_ref, job_config=job_config)
  job.result()

  table = client.get_table(table_ref)
  return table.num_rows


def upload_target_dataset_to_bq(config_path: str = "params.yaml") -> None:
  """Discovers the active sub-segment tag and stages data uploads to the warehouse.

  Args:
      config_path: Path to the YAML configuration file.

  Raises:
      ValueError: If BigQuery project or dataset configuration is missing.
  """
  params = utils.load_run_params(config_path)
  tag = utils.get_current_tag(params)

  cloud = params.get("cloud", {})
  project_id = cloud.get("project_id")
  dataset_id = cloud.get("dataset_id")

  if not project_id or not dataset_id:
    raise ValueError(
        "Mandatory BigQuery parameters 'cloud.project_id' or"
        " 'cloud.dataset_id' missing from configuration."
    )

  data_config = params.get("data", {})
  in_folder = data_config.get("clean_folder", "data/clean_data")
  in_csv = os.path.join(in_folder, f"{tag}.csv")

  if not os.path.exists(in_csv):
    logging.error(
        "Missing required input file %s. Aborting BigQuery upload.", in_csv
    )
    return

  out_dir = data_config.get("upload_folder", "data/upload_to_bq")
  os.makedirs(out_dir, exist_ok=True)
  marker_file = os.path.join(out_dir, f"{tag}_hash.txt")

  client = bigquery.Client(project=project_id)

  with open(in_csv, "rb") as source_file:
    file_hash = calculate_file_md5(source_file)
    num_rows = execute_bigquery_load(
        client, source_file, project_id, dataset_id, tag
    )

  logging.info(
      "Loaded %d rows into %s.%s.%s", num_rows, project_id, dataset_id, tag
  )

  logging.info("Writing output hash marker to %s", marker_file)
  with open(marker_file, "w") as f:
    f.write(file_hash)

  utils.upload_to_gcs(local_filepath=marker_file)
  logging.info(
      "Successfully uploaded dataset segment exclusively for tag: %s", tag
  )


if __name__ == "__main__":
  upload_target_dataset_to_bq()
