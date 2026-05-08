#!/bin/bash
set -e
echo "[sovereign-data-warehouse] Setting up..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete."
