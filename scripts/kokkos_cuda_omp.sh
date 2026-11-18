#!/bin/bash
# Kokkos 5.0.99 (/u/uli/kokkos-src) with CUDA 13.0 and OpenMP, the CUDA-only install's options plus OpenMP, into
# /u/uli/kokkos-cuda-omp: the one-build-two-backends toolchain of docs/ideas.md ("CPU and GPU together").
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/kokkos_cuda_omp.sh
#SBATCH -J kokkos_cuda_omp
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=01:00:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
build=/u/uli/kokkos-build-cuda-omp
rm -rf "$build"
cmake -S /u/uli/kokkos-src -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
    -DCMAKE_INSTALL_PREFIX=/u/uli/kokkos-cuda-omp -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=OFF \
    -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_DEPRECATED_CODE_5=ON \
    -DKokkos_ENABLE_COMPLEX_ALIGN=ON -DKokkos_ENABLE_TESTS=OFF -DKokkos_ENABLE_EXAMPLES=OFF >"$build.configure.log" 2>&1
cmake --build "$build" -j 18 >"$build.build.log" 2>&1
cmake --install "$build" >/dev/null
grep -h "set(Kokkos_DEVICES" /u/uli/kokkos-cuda-omp/lib64/cmake/Kokkos/KokkosConfig*.cmake
