#!/bin/bash
# peclet-voro for the library benchmark: its pinned Kokkos (5.1.1, PECLET_KOKKOS_TAG of cmake/PecletDeps.cmake) for
# CUDA (A100, sm_80) and for OpenMP (Ice Lake), the library's sibling header repos (peclet-core v1.3.0, peclet-morton
# v1.0.1, the tags PecletDeps.cmake pins) and the driver bench/libraries/peclet_voro.cpp against each.
# The library is header-only on this path: nothing of it is compiled except through the driver.
#   fetch (login node, downloads only):  bench/libraries/build_peclet_voro.sh fetch
#   build (Slurm):                       sbatch -o /raven/u/uli/libbench-data/logs/build-peclet-voro-%j.out \
#                                               bench/libraries/build_peclet_voro.sh
#SBATCH -J build_peclet_voro
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
set -euo pipefail
PV=/raven/u/uli/libbench-src/peclet-voro
VORO_COMMIT=d7371c92dacb7ec2c8225ec34d2d9184bb029f86  # v1.0.4-1-gd7371c9, main on 2026-10-01
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "${SLURM_SUBMIT_DIR:-}" ] && here="$SLURM_SUBMIT_DIR/bench/libraries"

if [ "${1:-}" = fetch ]; then
    mkdir -p "$PV"
    cd "$PV"
    [ -d voro ] || git clone https://github.com/computational-chemical-engineering/peclet-voro.git voro
    git -C voro checkout "$VORO_COMMIT"
    [ -d core ] || git clone --branch v1.3.0 https://github.com/computational-chemical-engineering/peclet-core.git core
    [ -d morton ] || git clone --branch v1.0.1 https://github.com/computational-chemical-engineering/peclet-morton.git morton
    [ -d kokkos-src ] || git clone --depth 1 --branch 5.1.1 https://github.com/kokkos/kokkos.git kokkos-src
    exit 0
fi

source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
jobs=${SLURM_CPUS_PER_TASK:-16}

# Kokkos 5.1.1: CUDA (+Serial host) and OpenMP (+Serial).
kokkos() {  # <name> <cmake args...>
    local name=$1
    shift
    [ -f "$PV/install/kokkos-$name/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && return
    cmake -S "$PV/kokkos-src" -B "$PV/build/kokkos-$name" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX="$PV/install/kokkos-$name" -DKokkos_ENABLE_SERIAL=ON "$@"
    cmake --build "$PV/build/kokkos-$name" -j "$jobs" --target install
}
kokkos cuda -DCMAKE_CXX_COMPILER="$PV/kokkos-src/bin/nvcc_wrapper" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
kokkos openmp -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ICX=ON

# The driver's CMake project (the library ships none for an external consumer of its device headers; this mirrors
# its tests/kokkos/CMakeLists.txt: same include dirs and definitions).
mkdir -p "$PV/driver"
cat > "$PV/driver/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(peclet_voro_libbench LANGUAGES CXX)
find_package(Kokkos 5 REQUIRED CONFIG)
add_executable(peclet_voro ${LIBBENCH_DIR}/peclet_voro.cpp)
target_include_directories(peclet_voro PRIVATE ${LIBBENCH_DIR} ${PV}/voro/include ${PV}/core/include ${PV}/morton/include)
target_compile_definitions(peclet_voro PRIVATE MORTON_ENABLE_KOKKOS=1 PECLET_CORE_HAVE_MORTON=1)
if(Kokkos_ENABLE_CUDA)
  target_compile_definitions(peclet_voro PRIVATE LIBBENCH_CUDA=1)
endif()
target_link_libraries(peclet_voro PRIVATE Kokkos::kokkos)
target_compile_features(peclet_voro PRIVATE cxx_std_20)
EOF
for backend in cuda openmp; do
    cxx=g++
    [ $backend = cuda ] && cxx="$PV/kokkos-src/bin/nvcc_wrapper"
    cmake -S "$PV/driver" -B "$PV/build/driver-$backend" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$cxx" \
        -DCMAKE_PREFIX_PATH="$PV/install/kokkos-$backend" -DLIBBENCH_DIR="$here" -DPV="$PV"
    cmake --build "$PV/build/driver-$backend" -j "$jobs" --verbose
    mkdir -p "$PV/install/bin"
    cp "$PV/build/driver-$backend/peclet_voro" "$PV/install/bin/peclet_voro_$backend"
done
ls -l "$PV/install/bin"
