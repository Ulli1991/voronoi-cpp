#!/bin/bash
# Basselin et al. GPU-Restricted-Power-Diagrams (OpenCL) for the library benchmark: fetch commit 196e7d7 of
# github.com/basselin7u/GPU-Restricted-Power-Diagrams into /raven/u/uli/libbench-src/basselin/src, apply basselin.patch
# (if any), configure with the code's CMake (-DUSE_DOUBLE=ON: writes cvoro_config.h with the paths of the .cl kernels,
# which are compiled at run time by the OpenCL driver), build its test_voronoi as a check, and the driver basselin.cpp
# with the library sources into install/bin/basselin (g++ -O3 -fopenmp; OpenCL ICD loader and CUDA runtime, the latter
# only for the device memory sampler).
#   sbatch -o /raven/u/uli/libbench-data/logs/build_basselin-%j.out bench/libraries/build_basselin.sh
#SBATCH -J build_basselin
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:15:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12.6 cmake
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$here/libbench.hpp" ] || here=/raven/u/uli/voronoi-cpp-libbench/bench/libraries
root=/raven/u/uli/libbench-src/basselin
commit=196e7d79125fb7edc75ed7b8f13990a7877c48f3
[ -d "$root/src/.git" ] || git clone https://github.com/basselin7u/GPU-Restricted-Power-Diagrams.git "$root/src"
cd "$root/src"
[ "$(git rev-parse HEAD)" = "$commit" ] || git checkout -q "$commit"
if git diff --quiet && [ -s "$here/basselin.patch" ]; then git apply "$here/basselin.patch"; fi
mkdir -p "$root/build" "$root/install/bin"
# CMAKE_POLICY_VERSION_MINIMUM: the code asks for cmake 2.8, which CMake 4 refuses (build job 32226351 failed so).
cmake -S "$root/src" -B "$root/build" -DCMAKE_BUILD_TYPE=Release -DUSE_DOUBLE:BOOL=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOpenCL_INCLUDE_DIR="$CUDA_HOME/include" -DOpenCL_LIBRARY="$CUDA_HOME/lib64/libOpenCL.so"
make -C "$root/build" -j16 test_voronoi
git status --short
s="$root/src"
g++ -O3 -std=c++17 -fopenmp -DCL_TARGET_OPENCL_VERSION=120 -DLIBBENCH_CUDA -I"$s" -I"$here" -I"$CUDA_HOME/include" \
    "$here/basselin.cpp" "$s/Grid.cpp" "$s/knearests.cpp" "$s/openCL.cpp" "$s/voronoi.cpp" "$s/Options.cpp" \
    "$s/Utils.cpp" -o "$root/install/bin/basselin" -L"$CUDA_HOME/lib64" -lOpenCL -lcudart \
    -Wl,-rpath,"$CUDA_HOME/lib64"
# OpenCL devices visible in this job
"$root/build/test_voronoi" --help 2>&1 | head -5 || true
"$root/install/bin/basselin" 2>&1 | head -2 || true
clinfo -l 2>/dev/null || true
ls -la "$root/install/bin"
