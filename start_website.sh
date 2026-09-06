#!/bin/bash
# ==============================================================================
# AirAssist PD - Interactive Web Application Launch Script
# ==============================================================================

PORT=${1:-5050}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "======================================================================"
echo "  Starting Air-Writing & Parkinson's Motor Analysis Web Dashboard"
echo "  URL: http://localhost:$PORT"
echo "======================================================================"

export PYTHONPATH="/opt/homebrew/Caskroom/miniconda/base/envs/physioassist/lib/python3.11/site-packages:$DIR"
export MPLCONFIGDIR="/tmp"

/usr/local/bin/python3 app/web_app.py --port "$PORT" --host 0.0.0.0
