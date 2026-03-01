#!/bin/bash
set -e

if ! command -v uv &> /dev/null
then
    echo "UV is not installed. Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

cd "$(dirname "$0")"

uv venv
uv sync
uv run mkdocs build --strict

echo "Documentation built successfully in the 'site' directory."
echo "To preview, run: uv run mkdocs serve"
