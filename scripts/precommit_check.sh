#!/bin/bash
set -ex

echo "Run Pre-Commit Check"
uv run pre-commit run --all-files
