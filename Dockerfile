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

# Use official lightweight Python 3.12 slim image
FROM python:3.12-slim

# Allow immediate print/logging output in container logs
ENV PYTHONUNBUFFERED=1

# Install OS level dependencies required for compiling c-extensions, git/dvc & gcloud storage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv lightning-fast lightweight package manager binary directly
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory explicitly
WORKDIR /app

# Inject PYTHONPATH permanently into the container runtime environment for flawless subpackage resolution
ENV PYTHONPATH=/app/src:/app

# Setup Python requirements
COPY requirements.txt .

# Install complete execution dependencies via uv into system python environment
RUN uv pip install --system --no-cache -r requirements.txt

# Disable git checks safely after DVC is installed
RUN dvc config --global core.no_scm true

# Copy entire codebase components
COPY . /app

# Force initialize a clean, isolated DVC repository inside the container context
RUN dvc init --no-scm -f

# Start the dedicated batch execution entrypoint directly using system python
ENTRYPOINT ["python", "cloud_run_job_main.py"]
