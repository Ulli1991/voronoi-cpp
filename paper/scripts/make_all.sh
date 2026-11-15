#!/bin/bash
# Redraws every paper figure from paper/data: bash paper/scripts/make_all.sh [python]. Extraction (needs job output and
# the library-benchmark volumes): extract_sweep.py, extract_stages.py, extract_oracle.py (see paper/data/SOURCES.md).
set -euo pipefail
py="${1:-python3}"  # matplotlib >= 3.8, numpy, pypdf (on Raven: /u/uli/jax-env/bin/python3)
cd "$(dirname "$0")"
for f in fig_walkthrough.py fig_library.py fig_tried.py fig_oracle.py fig_ceiling.py fig_sweep.py fig_stages.py fig_memory.py fig_resort.py fig_vortex.py; do
    [[ -f "$f" ]] && "$py" "$f"
done
