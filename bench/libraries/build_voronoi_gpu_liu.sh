#!/bin/bash
# Liu et al. 3D-Voronoi-GPU for the library benchmark: fetch commit b2c64f5 of github.com/xh-liu-tech/3D-Voronoi-GPU
# into /raven/u/uli/libbench-src/voronoi-gpu-liu/src, apply voronoi-gpu-liu.patch (PRESET settable, the two
# over-allocations in kn_prepare fixed as for VoroGPU), build the driver voronoi_gpu_liu.cu (which compiles upstream
# voronoi.cu into itself) with knearests.cu and extern/kNN-CUDA/knncuda.cu (referenced by voronoi.cu, not called
# by the driver; its legacy cuBLAS calls need -lcublas): install/bin/voronoi_gpu_liu (PRESET 0, the upstream default: K 90). -O3, sm_80, float as upstream.
#   sbatch -o /raven/u/uli/libbench-data/logs/build_voronoi_gpu_liu-%j.out bench/libraries/build_voronoi_gpu_liu.sh
#SBATCH -J build_liu
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
root=/raven/u/uli/libbench-src/voronoi-gpu-liu
commit=b2c64f527017fcd8877397dedb077f39d2ea20b4
[ -d "$root/src/.git" ] || git clone https://github.com/xh-liu-tech/3D-Voronoi-GPU.git "$root/src"
cd "$root/src"
[ "$(git rev-parse HEAD)" = "$commit" ] || git checkout -q "$commit"
if git diff --quiet && [ -s "$here/voronoi-gpu-liu.patch" ]; then git apply "$here/voronoi-gpu-liu.patch"; fi
git status --short
mkdir -p "$root/install/bin"
nvcc --version | tail -2
flags=(-O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -lineinfo -DPRESET=0 -Xcompiler -fopenmp,-O3
       -I"$root/src" -I"$here")
nvcc "${flags[@]}" "$here/voronoi_gpu_liu.cu" "$root/src/knearests.cu" "$root/src/extern/kNN-CUDA/knncuda.cu" \
    -o "$root/install/bin/voronoi_gpu_liu" -lgomp -lcublas
ls -la "$root/install/bin"
