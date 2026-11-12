#!/bin/bash
# votess for the library benchmark. SYCL toolchains tried, in this order:
#  1. Intel oneAPI icpx 2025.3 (module intel/2025.3): sycl-ls on an A100 node; the Raven install has no CUDA adapter
#     (no libur_adapter_cuda.so; the Codeplay "oneAPI for NVIDIA GPUs" plugin cannot be downloaded from Raven:
#     developer.codeplay.com does not resolve). Builds the driver for votess's CPU path (device::cpu).
#  2. AdaptiveCpp v25.02.0 against LLVM 18 (/mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8) and CUDA 12.2: generic SSCP target (PTX at run time),
#     the driver built with acpp for votess's SYCL device path on the A100.
#   fetch (login node):  bench/libraries/build_votess.sh fetch
#   build (Slurm):       sbatch -o /raven/u/uli/libbench-data/logs/build-votess-%j.out bench/libraries/build_votess.sh
#SBATCH -J build_votess
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
set -uo pipefail
V=/raven/u/uli/libbench-src/votess
COMMIT=f0f805fd30a77975cc9d690502d9941f58441cdf  # main 2025-07-15
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "${SLURM_SUBMIT_DIR:-}" ] && here="$SLURM_SUBMIT_DIR/bench/libraries"
if [ "${1:-}" = fetch ]; then
    mkdir -p "$V" && cd "$V"
    [ -d votess-src ] || git clone https://github.com/samridh-dev/votess.git votess-src
    git -C votess-src checkout "$COMMIT"
    [ -d acpp-src ] || git clone --depth 1 --branch v25.02.0 https://github.com/AdaptiveCpp/AdaptiveCpp.git acpp-src
    exit 0
fi
jobs=${SLURM_CPUS_PER_TASK:-16}
inc="-I$V/votess-src/include -I$V/votess-src/src -I$here"
mkdir -p "$V/install/bin"
source /etc/profile.d/modules.sh

echo "=== 1. oneAPI icpx"
module purge; module load intel/2025.3
icpx --version | head -1
sycl-ls 2>&1 | sed 's/^/sycl-ls: /'
icpx -fsycl -O3 -std=c++17 -fopenmp -pthread $inc "$here/votess.cpp" -o "$V/install/bin/votess_icpx" && echo "icpx driver built"

echo "=== 2. AdaptiveCpp"
module purge; module load gcc/13; module load cuda/12.2; module load cmake; module load boost/1.83  # boost.context/fiber: AdaptiveCpp CPU backend
export PATH=/mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8/bin:$PATH LD_LIBRARY_PATH=/mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8/lib:${LD_LIBRARY_PATH:-}
export CUDA_PATH="$(dirname "$(dirname "$(which nvcc)")")"
# clang 18 would otherwise take the system GCC 7.5 headers and libstdc++; use gcc/13's.
GCCTC="--gcc-toolchain=$(dirname "$(dirname "$(which gcc)")")"
if [ ! -x "$V/install/acpp/bin/acpp" ]; then
    cmake -S "$V/acpp-src" -B "$V/build/acpp" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$V/install/acpp" \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS="$GCCTC" -DCMAKE_CXX_FLAGS="$GCCTC" \
        -DCMAKE_EXE_LINKER_FLAGS="$GCCTC" -DCMAKE_SHARED_LINKER_FLAGS="$GCCTC" \
        -DBOOST_ROOT="$BOOST_ROOT" -DLLVM_DIR="$(llvm-config --cmakedir)" -DWITH_CUDA_BACKEND=ON -DCUDAToolkit_ROOT="$CUDA_PATH" \
        -DWITH_ROCM_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF \
        && cmake --build "$V/build/acpp" -j "$jobs" --target install
fi
"$V/install/acpp/bin/acpp-info" -l 2>&1 | sed 's/^/acpp-info: /'
# VOTESS_DEVICE_PEAK, not LIBBENCH_CUDA: cuda_runtime_api.h's dim3 clashes with AdaptiveCpp's (see votess.cpp).
"$V/install/acpp/bin/acpp" $GCCTC -O3 -std=c++17 -fopenmp -DVOTESS_DEVICE_PEAK $inc "$here/votess.cpp" \
    -L"$CUDA_PATH/lib64" -lcudart -Wl,-rpath,"$CUDA_PATH/lib64" -o "$V/install/bin/votess_acpp" && echo "acpp driver built"
ls -l "$V/install/bin"
