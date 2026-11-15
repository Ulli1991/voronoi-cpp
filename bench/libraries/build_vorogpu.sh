#!/bin/bash
# VoroGPU for the library benchmark: fetch commit f4d7e7e of github.com/basselin7u/VoroGPU into
# /raven/u/uli/libbench-src/vorogpu/src, apply vorogpu.patch (no-tetrahedron path = the paper's box algorithm, the
# missing stopwatch.h, PRESET settable, two over-allocations in kn_prepare fixed), build the driver vorogpu.cu (which
# compiles upstream voronoi.cu into itself) with knearests.cu, one binary per preset: install/bin/vorogpu_k190
# (PRESET 0, white noise), vorogpu_k90 (PRESET 1, perturbed grid) and vorogpu_k35 (PRESET 2, blue noise, the upstream default). -O3, sm_80, float as upstream. The CPU fallback (geogram) of the paper is
# not in the repository (CMakeLists names CPU/*.cpp that are absent), so it is not built.
#   sbatch -o /raven/u/uli/libbench-data/logs/build_vorogpu-%j.out bench/libraries/build_vorogpu.sh
#SBATCH -J build_vorogpu
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:15:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12.6
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$here/libbench.hpp" ] || here=/raven/u/uli/voronoi-cpp-libbench/bench/libraries
root=/raven/u/uli/libbench-src/vorogpu
commit=f4d7e7e40cbe03ccc4f40b5351df998a7def9acc
[ -d "$root/src/.git" ] || git clone https://github.com/basselin7u/VoroGPU.git "$root/src"
cd "$root/src"
[ "$(git rev-parse HEAD)" = "$commit" ] || git checkout -q "$commit"
if git diff --quiet && [ ! -e stopwatch.h ] && [ -s "$here/vorogpu.patch" ]; then git apply "$here/vorogpu.patch"; fi
git status --short
mkdir -p "$root/install/bin"
nvcc --version | tail -2
for preset in 0 1 2; do  # the README's settings: white noise, perturbed grid, blue noise
    k=$(case $preset in 0) echo 190 ;; 1) echo 90 ;; 2) echo 35 ;; esac)
    flags=(-O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -lineinfo -DPRESET=$preset -Xcompiler -fopenmp,-O3
           -I"$root/src" -I"$here")
    nvcc "${flags[@]}" "$here/vorogpu.cu" "$root/src/knearests.cu" -o "$root/install/bin/vorogpu_k$k" -lgomp &
done
wait
ls -la "$root/install/bin"
