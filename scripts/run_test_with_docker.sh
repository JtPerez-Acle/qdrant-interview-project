#!/bin/bash
# Simple script to run the test pipeline with Docker permissions

# Get the absolute path to the Python interpreter in the current environment
PYTHON_PATH=$(which python)

# Run the test pipeline with sudo using the correct Python interpreter
sudo PYTHONPATH=. $PYTHON_PATH scripts/test_pipeline.py "$@"
