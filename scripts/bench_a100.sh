#!/bin/bash
# The A100 checks of TESTING.md and the standard measurements. Submit from
# the repository root with a built CUDA directory; logs must be on a shared
# file system:
#   sbatch -o build-cuda/bench-%j.out scripts/bench_a100.sh build-cuda
#SBATCH -J voronoi_bench
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
build="${1:-build-cuda}"
nvidia-smi -L
ctest --test-dir "$build" --output-on-failure
for kind in jittered uniform clustered halo lattice wave; do
    "$build/voronoi_bench" rebuild 2097152 "$kind" warp 3
done
for kind in jittered uniform clustered halo lattice wave; do
    "$build/voronoi_bench" step 262144 "$kind" 0.3 3 --verify | grep -E "^start|verify"
done
for d in 0.03 0.1 0.3; do
    "$build/voronoi_bench" step 2097152 jittered "$d" 10
done
