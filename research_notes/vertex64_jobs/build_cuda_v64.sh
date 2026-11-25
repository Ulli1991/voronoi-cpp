#!/bin/bash
#SBATCH -p interactive
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH -J v64-cuda
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/build_cuda_v64-%j.out
# #64: rebuild the CUDA diagnostic build (and the OpenMP one) after a change to the diagnostic.
set -e
cd /raven/u/uli/voronoi-cpp-vertex64
module purge; module load gcc/14 cuda/13.0 cmake
cmake --build build-cuda-v64 -j8 --target voronoi_bench mesh_fuzz_test 2>&1 | grep -E " error|Error|Built" | head -40
module purge; module load gcc/15 cmake
cmake --build build-omp-v64 -j8 --target voronoi_bench mesh_fuzz_test 2>&1 | grep -E " error|Error|Built" | head -40
