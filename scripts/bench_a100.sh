#!/bin/bash
# Standard A100 measurements and the CUDA tests. Submit from the repository
# root with a built CUDA directory; logs must be on a shared file system:
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
for kind in jittered uniform; do
    "$build/voronoi_bench" rebuild 2097152 "$kind" warp 3
done
"$build/voronoi_bench" step 2097152 jittered 0.03 10
"$build/voronoi_bench" step 262144 jittered 0.3 10 --verify
ctest --test-dir "$build" --output-on-failure
