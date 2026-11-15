#!/bin/bash
# Several library runs in one A100 job (gpudev: few jobs per user, 15 minutes), so the libraries of one job ran on the
# same GPU. Each item is "<run script> <ENV=value,...>"; ENV values may hold spaces in quotes. Records of every item go
# to this job's stdout. Each chain starts with a reference run of voronoi-cpp's warp rebuild on the same inputs.
#   sbatch -o bench/libraries/results/chain-a100-%j.jsonl -e <log> bench/libraries/run_chain_a100.sh <data> <cuda build> \
#       "run_gdel3d_a100.sh SIZES=2097152" "run_votess_a100.sh SIZES=2097152" ...
#SBATCH -J libbench_chain
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
here="$SLURM_SUBMIT_DIR/bench/libraries"
data="$1"
build="$2"
shift 2
nvidia-smi -L >&2
for item in "$@"; do
    script=${item%% *}
    env_part=${item#"$script"}
    echo "# $(date +%T) chain item: $item" >&2
    (
        IFS=','
        for kv in $env_part; do
            kv="${kv# }"
            [ -n "$kv" ] && export "$kv"
        done
        unset IFS
        if [ "$script" = run_voronoi_cpp_a100.sh ]; then
            bash "$here/$script" "$build" "$data"
        else
            bash "$here/$script" "$data"
        fi
    ) || echo "# chain item failed: $item (exit $?)" >&2
done
echo "# $(date +%T) chain done" >&2
