#!/bin/bash
# Submits the weak-scaling sweep in 15-minute pieces on gpu1:  bash paper/scripts/submit_sweep.sh <bin dir> [data dir]
# Logs: paper/data/raw/weak-<job>.jsonl (records), paper/logs/weak-<job>.err; then python3 paper/scripts/extract.py.
set -euo pipefail
bin="$1" data="${2:-/ptmp/uli/voronoi-paper}"
cd "$(git rev-parse --show-toplevel)"
mkdir -p paper/data/raw paper/logs
go() { sbatch --parsable -o 'paper/data/raw/weak-%j.jsonl' -e 'paper/logs/weak-%j.err' --export="ALL,$1" \
    paper/scripts/weak_scaling_a100.sh "$bin" "$data"; }
go "SIZES=32768 262144 2097152"
go "SIZES=16777216,KINDS=jittered bluenoise uniform"
go "SIZES=16777216,KINDS=clustered halo"
go "SIZES=16777216,KINDS=lattice wave"
go "SIZES=2097152,PATHS=none,SPACINGS=0.01 1"
