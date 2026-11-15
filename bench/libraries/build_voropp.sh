#!/bin/bash
# voro++ for the library benchmark: master (serial; container_periodic) and the dev branch (multithreaded extension of
# Lu, Lazar, Rycroft, CPC 2023) of github.com/chr1shr/voro, each built as a static library with gcc 14 -O3
# -march=native (dev: -fopenmp), installed under /raven/u/uli/libbench-src/voropp/install/{master,dev}, and the
# drivers voropp_bench (master) and voropp_mt_bench (dev) in install/bin.
#   sbatch -o /raven/u/uli/libbench-data/logs/build_voropp-%j.out bench/libraries/build_voropp.sh
#SBATCH -J libbench_build_voropp
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
root=/raven/u/uli/libbench-src/voropp
master=b0dac575a47af0f90b5b100e6dc199a493c7cb83
dev=61099f682c6672a275b6be7de5923f2a1a7bf05e
flags="-O3 -march=native"
[ -d "$root/src" ] || git clone https://github.com/chr1shr/voro "$root/src"
git -C "$root/src" fetch -q origin
git -C "$root/src" checkout -q "$master"
[ -d "$root/src-dev" ] || git -C "$root/src" worktree add --detach "$root/src-dev" "$dev"
git -C "$root/src-dev" checkout -q "$dev"
[ -f "$root/src-dev/config.mk" ] || cp "$root/src-dev/config.mk.template" "$root/src-dev/config.mk"
build() {  # <source tree> <install dir> <flags>
    make -C "$1/src" clean >/dev/null
    make -C "$1/src" -j"${JOBS:-16}" libvoro++.a CXX=g++ CFLAGS="$3"
    mkdir -p "$2/include" "$2/lib"
    cp "$1"/src/*.hh "$2/include/"
    cp "$1/src/libvoro++.a" "$2/lib/"
}
build "$root/src" "$root/install/master" "$flags"
build "$root/src-dev" "$root/install/dev" "$flags -fopenmp"
mkdir -p "$root/install/bin"
g++ -std=c++17 $flags -I"$root/install/master/include" "$here/voropp.cpp" -L"$root/install/master/lib" -lvoro++ \
    -o "$root/install/bin/voropp_bench"
g++ -std=c++17 $flags -fopenmp -DVOROPP_DEV -I"$root/install/dev/include" "$here/voropp.cpp" \
    -L"$root/install/dev/lib" -lvoro++ -o "$root/install/bin/voropp_mt_bench"
g++ --version | head -1
ls -l "$root/install/bin"
