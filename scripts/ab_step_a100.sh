#!/bin/bash
# Two builds of voronoi_bench on one A100, alternating: step rates and checksums (docs/measurements.md, step kernels).
#   sbatch -o build-cuda/jobs/%x-%j.out scripts/ab_step_a100.sh <dir> <tag of build A> <tag of build B> <mode>
# mode: rates128, rates256 (128³ or 256³, three spacings, two rounds each) or bits (checksums, 128³).
# <dir>/voronoi_bench_<tag> are the binaries (copied there, so rebuilds do not race the job).
#SBATCH -J ab_step
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
dir="$1" a="$2" b="$3" mode="$4"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
run() {  # run <tag> <args...>: one bench run, its lines prefixed by the tag
    local tag="$1"
    shift
    "$dir/voronoi_bench_$tag" "$@" | grep -E "^(gpuflips|geometry_checksum|mesh)" | sed "s/^/$tag $* | /"
}
if [[ "$mode" == rates* ]]; then
    n=2097152
    [[ "$mode" == rates256 ]] && n=16777216
    for d in 0.03 0.1 0.3; do
        for rep in 1 2; do
            run "$a" step "$n" jittered "$d" 10
            run "$b" step "$n" jittered "$d" 10
        done
    done
fi
if [[ "$mode" == bits ]]; then
    for kind in jittered clustered; do
        for d in 0.03 0.1 0.3; do
            run "$a" step 2097152 "$kind" "$d" 5 --checksum
            run "$b" step 2097152 "$kind" "$d" 5 --checksum
        done
    done
fi
nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader
