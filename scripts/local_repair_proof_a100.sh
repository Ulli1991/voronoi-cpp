#!/bin/bash
# The whole S2 proof in one gpu1 job: builds main (ref, baseline worktree) and the branch, then A100 ctest, the
# verified cases, the jumper rates at 128³ and 256³, the default path's checksums and rates, and CI's host steps:
#   sbatch -o build-cuda/gpu-run/proof-%j.txt scripts/local_repair_proof_a100.sh <baseline worktree>
#SBATCH -J local_repair_proof
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=08:00:00
set -uo pipefail
base="$1"
here=$(pwd)
bin=build-cuda/bin-$SLURM_JOB_ID
(
    source /etc/profile.d/modules.sh
    module purge
    module load gcc/14 cuda/13.0 cmake
    echo "== build: ref $(git -C "$base" log --oneline -1), new $(git log --oneline -1)"
    cmake --build "$base/build-cuda" --parallel 18 --target voronoi_bench 2>&1 | grep -E " error|error:" | head
    cmake --build build-cuda --parallel 18 2>&1 | grep -E " error|error:" | head
    mkdir -p "$bin"
    cp "$base/build-cuda/voronoi_bench" "$bin/voronoi_bench_ref"
    cp build-cuda/voronoi_bench "$bin/voronoi_bench_new"
    echo "== ctest"
    ctest --test-dir build-cuda --output-on-failure 2>&1 | tail -40
)
part() {  # part <name> <n> [steps]
    echo "== part $1 $2"
    STEPS=${3:-} bash scripts/local_repair_a100.sh "$bin" "$1" "$2"
}
part verify 2097152 6
part speed 2097152 12
part speed 16777216 8
part smooth 2097152 10
echo "== ci host steps"
cd "$here" && bash scripts/ci_host_general.sh origin/main src/bench/voronoi_bench.cpp src/test/mesh_cases/fallback.cpp \
    src/test/repair_test.cpp src/test/modes_test.cpp
