#!/bin/bash
# gDel3D for the library benchmark: fetch commit a429be8 of github.com/ashwin/gDel3D into
# /raven/u/uli/libbench-src/gdel3d/src, apply gdel3d.patch (the port to CUDA 12 / sm_80), build the library with nvcc
# into install/lib/libgdel3d.a and the driver gdel3d.cu into install/bin/gdel3d. Double precision (REAL_TYPE_FP32 not
# defined), -O3, sm_80.
#   sbatch -o /raven/u/uli/libbench-data/logs/build_gdel3d-%j.out bench/libraries/build_gdel3d.sh
#SBATCH -J build_gdel3d
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:15:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12.6
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$here/libbench.hpp" ] || here=/raven/u/uli/voronoi-cpp-libbench/bench/libraries
root=/raven/u/uli/libbench-src/gdel3d
commit=a429be8d702a1f0f0d9366e14f7f492b1c071c9b
if [ ! -d "$root/src/.git" ]; then
    git clone https://github.com/ashwin/gDel3D.git "$root/src"
fi
cd "$root/src"
[ "$(git rev-parse HEAD)" = "$commit" ] || git checkout -q "$commit"
# A clean tree gets the patch; a patched tree is left as it is.
if git diff --quiet && [ -s "$here/gdel3d.patch" ]; then git apply "$here/gdel3d.patch"; fi
git diff --stat
src="$root/src/GDelFlipping/src"
mkdir -p "$root/build" "$root/install/lib" "$root/install/bin"
nvcc --version | tail -2
flags=(-O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -Xcompiler -fopenmp,-O3 -I"$src" -I"$src/gDel3D")
objs=()
for f in gDel3D/GpuDelaunay.cu gDel3D/GPU/KerDivision.cu gDel3D/GPU/KerPredicates.cu gDel3D/GPU/ThrustWrapper.cu \
         gDel3D/CPU/predicates.cpp gDel3D/CPU/PredWrapper.cpp gDel3D/CPU/Splaying.cpp gDel3D/CPU/Star.cpp; do
    o="$root/build/$(basename "${f%.*}").o"
    nvcc "${flags[@]}" -c "$src/$f" -o "$o" &
    objs+=("$o")
done
wait
rm -f "$root/install/lib/libgdel3d.a"
ar rcs "$root/install/lib/libgdel3d.a" "${objs[@]}"
nvcc "${flags[@]}" -I"$here" "$here/gdel3d.cu" "$root/install/lib/libgdel3d.a" -o "$root/install/bin/gdel3d" -lgomp
echo "built $root/install/bin/gdel3d"
