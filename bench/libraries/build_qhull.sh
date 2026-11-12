#!/bin/bash
# Qhull 2020.2 (8.0.2) for the library benchmark: tag v8.0.2 of github.com/qhull/qhull (the current stable release;
# the 8.1 tags are alphas), CMake Release (-O3 -march=native), static reentrant library libqhullstatic_r installed
# into install/, and the driver qhull_bench in install/bin.
#   sbatch -o /raven/u/uli/libbench-data/logs/build_qhull-%j.out bench/libraries/build_qhull.sh
#SBATCH -J libbench_build_qhull
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cmake
here=/raven/u/uli/voronoi-cpp-libbench/bench/libraries
root=/raven/u/uli/libbench-src/qhull
tag=v8.0.2  # 613debeaea72ee66626dace9ba1a2eff11b5d37d
[ -d "$root/src" ] || git clone https://github.com/qhull/qhull "$root/src"
git -C "$root/src" checkout -q "$tag"
cmake -S "$root/src" -B "$root/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_FLAGS="-march=native" -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_INSTALL_PREFIX="$root/install" \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5  # CMake 4 refuses its cmake_minimum_required(3.0)
cmake --build "$root/build" -j16 --target qhullstatic_r
cmake --install "$root/build" 2>/dev/null || true  # installs headers and what was built
mkdir -p "$root/install/bin" "$root/install/lib"
lib=$(find "$root/build" -name 'libqhullstatic_r.a' | head -1)
cp "$lib" "$root/install/lib/"
mkdir -p "$root/install/include/libqhull_r"
cp "$root"/src/src/libqhull_r/*.h "$root/install/include/libqhull_r/"
g++ -std=c++17 -O3 -march=native -I"$root/install/include" "$here/qhull.cpp" -L"$root/install/lib" -lqhullstatic_r \
    -o "$root/install/bin/qhull_bench"
g++ --version | head -1
ls -l "$root/install/bin"
