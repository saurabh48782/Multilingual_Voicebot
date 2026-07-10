#!/bin/bash
set -ex

uv run pytest -vv tests/integration
