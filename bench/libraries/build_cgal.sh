#!/bin/bash
# CGAL for the library benchmark: CGAL 6.2.1 (header-only release tarball), oneTBB 2023.2.0 built from source,
# Boost 1.74 (module, headers only; CGAL 6 needs >= 1.74), GMP/MPFR from /usr; then the driver cgal_bench and a
# smoke test (P3DT and the padded parallel DT3 must give the same volumes on 4096 random points).
#   sbatch -o /raven/u/uli/libbench-data/logs/build_cgal-%j.out bench/libraries/build_cgal.sh
#SBATCH -J build_cgal
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:45:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cmake boost/1.74
here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/cgal.cpp" ] || here=/raven/u/uli/voronoi-cpp-libbench/bench/libraries
root=/raven/u/uli/libbench-src
cgal_version=6.2.1
tbb_version=2023.2.0

# CGAL: release tarball (sha256 as published in the release's sha256sum.txt).
mkdir -p "$root/cgal/dl"
cd "$root/cgal"
[ -f dl/CGAL-$cgal_version-library.tar.xz ] ||
    wget -q -P dl https://github.com/CGAL/cgal/releases/download/v$cgal_version/CGAL-$cgal_version-library.tar.xz
echo "8e7ce1e30e8ffcb3e69d153c3a561bb1f38bbb4becd8e6544cbaa0fb2889af62  dl/CGAL-$cgal_version-library.tar.xz" | sha256sum -c
[ -d CGAL-$cgal_version ] || tar xf dl/CGAL-$cgal_version-library.tar.xz
rm -rf install && mkdir -p install && ln -sfn "$root/cgal/CGAL-$cgal_version/include" install/include

# oneTBB from source.
mkdir -p "$root/tbb/dl"
cd "$root/tbb"
[ -f dl/oneTBB-$tbb_version.tar.gz ] ||
    wget -q -O dl/oneTBB-$tbb_version.tar.gz https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v$tbb_version.tar.gz
echo "80cb62e62defeee67ea1885a6f72d97ac81404f8fde9cded62fb51524808a30c  dl/oneTBB-$tbb_version.tar.gz" | sha256sum -c
if [ ! -f install/lib64/libtbb.so ] && [ ! -f install/lib/libtbb.so ]; then
    rm -rf oneTBB-$tbb_version build
    tar xf dl/oneTBB-$tbb_version.tar.gz
    cmake -S oneTBB-$tbb_version -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/tbb/install" \
        -DTBB_TEST=OFF -DTBB_EXAMPLES=OFF -DTBB_STRICT=OFF
    cmake --build build -j"${JOBS:-16}"
    cmake --install build
fi
tbblib="$root/tbb/install/lib64"
[ -d "$tbblib" ] || tbblib="$root/tbb/install/lib"

# The driver.
flags="-O3 -march=native -DNDEBUG -std=c++20 -fopenmp"
mkdir -p "$root/cgal/bin"
g++ $flags -DCGAL_LINKED_WITH_TBB -I"$root/cgal/CGAL-$cgal_version/include" -I"$BOOST_HOME/include" \
    -I"$root/tbb/install/include" "$here/cgal.cpp" -o "$root/cgal/bin/cgal_bench" \
    -L"$tbblib" -Wl,-rpath,"$tbblib" -ltbb -ltbbmalloc -lgmp -lmpfr -lpthread
echo "built $root/cgal/bin/cgal_bench with g++ $(g++ -dumpfullversion) $flags, CGAL $cgal_version," \
    "oneTBB $tbb_version, Boost $(basename "$BOOST_HOME")"

# Smoke test on 4096 random points of the 2^-30 grid.
t="$root/cgal/smoke"
mkdir -p "$t"
/raven/u/uli/jupyter_env/bin/python -I -c "
import numpy as np, sys
n=4096; r=np.random.default_rng(1); x=r.integers(0,2**30,size=(n,3))/2.0**30
with open(sys.argv[1],'wb') as f: np.array([n,1],dtype='<u8').tofile(f); x.astype('<f8').tofile(f)
" "$t/uniform_4096.pts"
"$root/cgal/bin/cgal_bench" p3dt "$t/uniform_4096.pts" --repeats=1 --volumes="$t/p3dt.f64"
"$root/cgal/bin/cgal_bench" dt3 "$t/uniform_4096.pts" --threads=16 --repeats=1 --volumes="$t/dt3.f64"
/raven/u/uli/jupyter_env/bin/python -I -c "
import numpy as np, sys
a=np.fromfile(sys.argv[1]); b=np.fromfile(sys.argv[2])
print('smoke: sums', a.sum()-1, b.sum()-1, 'max rel diff', np.max(np.abs(a-b)/a), 'min', a.min(), b.min())
" "$t/p3dt.f64" "$t/dt3.f64"
