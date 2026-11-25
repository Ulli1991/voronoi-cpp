#!/bin/bash
# Submits the throughput-against-problem-size sweep in 15-minute pieces on gpu1:  bash paper/scripts/submit_sweep.sh <bin dir> [data dir]
# Logs: paper/data/raw/sweep-<job>.jsonl (records), paper/logs/sweep-<job>.err; then python3 paper/scripts/extract.py.
set -euo pipefail
bin="$1" data="${2:-/ptmp/uli/voronoi-paper}"
cd "$(git rev-parse --show-toplevel)"
mkdir -p paper/data/raw paper/logs
go() { sbatch --parsable -o 'paper/data/raw/sweep-%j.jsonl' -e 'paper/logs/sweep-%j.err' --export="ALL,$1" \
    paper/scripts/throughput_vs_n_a100.sh "$bin" "$data"; }
go "SIZES=32768 262144 2097152"
go "SIZES=16777216,KINDS=jittered bluenoise uniform"
go "SIZES=16777216,KINDS=clustered halo"
go "SIZES=16777216,KINDS=lattice wave"
go "SIZES=2097152,PATHS=none,SPACINGS=0.01 1"
