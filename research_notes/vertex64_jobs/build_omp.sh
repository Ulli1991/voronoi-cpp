#!/bin/bash
#SBATCH -p interactive
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J v64-omp
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/build_omp-%j.out
# #64: OpenMP build with the diagnostic (gcc 15), then the #62 seeds with `one SEED`.
set -e
module purge; module load gcc/15 cmake
cd /raven/u/uli/voronoi-cpp-vertex64
cmake -S . -B build-omp-v64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-gcc15 \
  -DCMAKE_CXX_FLAGS=-march=icelake-server -DVORONOI_VERTEX64=ON > /dev/null
cmake --build build-omp-v64 -j8 --target mesh_fuzz_test voronoi_bench 2>&1 | grep -E "error|Error|warning|Built" | head -60
export OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=cores
out=/ptmp/uli/voronoi-cpp-vertex64/omp_seeds62
mkdir -p $out
for seed in 91001608214 91003117362 91003988275 91002082882 91003679558 91003836206 91002999916 91003158902 \
            92003949072 92002430746 92002596727 92001619850 92000341041 92001905473 92001421665 92002557534 92003651920; do
  echo "=== seed $seed"
  timeout 600 build-omp-v64/mesh_fuzz_test one $seed > $out/one_$seed.txt 2>&1 || echo "exit $?"
  grep -E "^outcome|^label|v64 counters" $out/one_$seed.txt
done
