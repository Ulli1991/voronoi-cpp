#!/bin/bash
# The re-sort measurements (#34) on one A100: sbatch -o build-cuda/jobs/%x-%j.out scripts/resort_a100.sh <mode> ...
#SBATCH -J resort
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
bin=build-cuda/jobs
mode="$1"
shift
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,memory.used --format=csv,noheader -l 5 >"$bin/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
case "$mode" in
decay)  # decay <binary> <n> <steps> <order every> [flags]: the smooth flow at 0.1 spacings, order metrics
    exe="$1" n="$2" steps="$3" every="$4"; shift 4
    "$bin/$exe" step "$n" jittered 0.1 "$steps" --order="$every" "$@" | grep -vE "^  "
    ;;
table)  # table <binary> <n> <steps> <flag sets separated by ';'>: one run per flag set, same job
    exe="$1" n="$2" steps="$3"
    IFS=';' read -ra sets <<<"$4"
    for flags in "${sets[@]}"; do
        echo "== run $flags"
        # shellcheck disable=SC2086
        "$bin/$exe" step "$n" jittered 0.1 "$steps" $flags | grep -vE "^  "
    done
    ;;
rebuild)  # rebuild <binary> <n>: rebuild rate
    "$bin/$1" rebuild "$2" jittered warp 5 --no-check
    ;;
run)  # run <binary> [args]: one program of build-cuda/jobs, such as a test
    exe="$1"; shift
    "$bin/$exe" "$@"
    ;;
*)
    echo "unknown mode $mode"
    exit 2
    ;;
esac
