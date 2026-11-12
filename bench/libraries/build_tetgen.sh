#!/bin/bash
# TetGen 1.6.0 for the library benchmark: tag v1.6.0 of github.com/TetGen/TetGen (the WIAS mirror of
# wias-berlin.de/software/tetgen), built as in its makefile (tetgen.cxx -O3, here also -march=native; predicates.cxx
# -O0 to keep Shewchuk's exact arithmetic exact) with -DTETLIBRARY into install/lib/libtet.a, and the driver
# tetgen_bench in install/bin.
#   sbatch -o /raven/u/uli/libbench-data/logs/build_tetgen-%j.out bench/libraries/build_tetgen.sh
#SBATCH -J libbench_build_tetgen
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14
here=/raven/u/uli/voronoi-cpp-libbench/bench/libraries
root=/raven/u/uli/libbench-src/tetgen
tag=v1.6.0  # 535f9c41f44abc832a7bbf2c9c7af003d1c18f3c
[ -d "$root/src" ] || git clone https://github.com/TetGen/TetGen "$root/src"
git -C "$root/src" checkout -q "$tag"
mkdir -p "$root/install/include" "$root/install/lib" "$root/install/bin" "$root/build"
cd "$root/build"
g++ -O3 -march=native -DTETLIBRARY -c "$root/src/tetgen.cxx" -o tetgen.o
g++ -O0 -DTETLIBRARY -c "$root/src/predicates.cxx" -o predicates.o
ar rcs "$root/install/lib/libtet.a" tetgen.o predicates.o
cp "$root/src/tetgen.h" "$root/install/include/"
g++ -std=c++17 -O3 -march=native -fopenmp -DTETLIBRARY -I"$root/install/include" "$here/tetgen.cpp" -L"$root/install/lib" -ltet \
    -o "$root/install/bin/tetgen_bench"
g++ --version | head -1
ls -l "$root/install/bin"
