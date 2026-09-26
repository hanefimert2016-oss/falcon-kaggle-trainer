#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest pyyaml
python -m py_compile falconctl.py kaggle/train_flm.py
pytest -q
printf '\nSetup complete. Configure KAGGLE_API_TOKEN and KAGGLE_OWNER before using Kaggle.\n'
