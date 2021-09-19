#!/usr/bin/env bash

# Copyright 2021 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

# As its name suggests, this is a bootstrap script. It is intended to be called by cron. Its sole purpose is to download
# the latest "tasks" git repository, then hand control over to generate_predictions.sh or generate_prescriptions.sh.

# This is split into a separate script to give us flexibility in updating the main scripts.
# This file needs to be deployed *manually* to the sandboxes.

# See https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html for what these do
set -o errexit
set -o errtrace
set -o nounset
set -o pipefail
set -o xtrace

# What mode is this sandbox running in? "prescribe" or "predict"
mode=$1

export TZ=":America/Los_Angeles"
echo "Running bootstrap at $(date)"

# Get latest from github as a zip file and extract it to user home directory

# Repo to download from
repo_name="covid-xprize-robotasks"

# Use this branch from Github
branch="main"

# Path to downloaded local repo
repo_dir="$HOME/$repo_name-$branch"

# Remove existing copy
rm -rf "$repo_dir"

# Get local copy of repo
archive_file="$HOME/$repo_name.tar.gz"
wget --quiet --output-document "$archive_file" https://github.com/leaf-ai/$repo_name/archive/$branch.tar.gz

# Unzip to destination directory
mkdir -p "$repo_dir" &&
  tar --overwrite --extract --directory "$HOME" --file "$archive_file" &&
  rm "$archive_file"

# Launch the main script. The script launched will depend on which mode we're running in -- "predictions"
# or "prescriptions"
if [ "$mode" = "predictions" ]; then
  main_script="$repo_dir"/judging/generate_predictions.sh
elif [ "$mode" = "prescriptions" ]; then
  main_script="$repo_dir"/judging/scripts/prescribe/generate_prescriptions.sh
else
  echo "Unknown mode: $mode" >&2
  exit 1
fi

chmod +x "$main_script"
$main_script "$repo_dir"