#!/bin/bash
#SBATCH -p interactive
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:59:00
#SBATCH -J v64-cuda
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/build_cuda-%j.out
# #64: CUDA builds of the worktree, default (bits and time reference) and with the diagnostic.
set -e
module purge; module load gcc/14 cuda/13.0 cmake
cd /raven/u/uli/voronoi-cpp-vertex64
for cfg in "build-cuda OFF" "build-cuda-v64 ON"; do
  set -- $cfg
  cmake -S . -B $1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DVORONOI_VERTEX64=$2 > /dev/null
  echo "=== $1"
  cmake --build $1 -j8 --target voronoi_bench mesh_fuzz_test 2>&1 | grep -E "error|Error|warning|Built" | head -60
done
