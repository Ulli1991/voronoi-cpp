#!/bin/bash
#SBATCH -p interactive
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J v64-fz
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/fuzz_cpu-%j.out
# #64: CPU fuzz (OpenMP diagnostic build, one thread per worker), counters per seed in RESULTS.v64.
# Profiles: small (4 workers, the #62 base 91000000001), long (2), medium (2); 20 minutes each.
module purge; module load gcc/15
cd /raven/u/uli/voronoi-cpp-vertex64
out=/ptmp/uli/voronoi-cpp-vertex64/fuzz_cpu_$SLURM_JOB_ID
mkdir -p $out
cp build-omp-v64/mesh_fuzz_test $out/fuzz
export OMP_NUM_THREADS=1
for w in 0 1 2 3; do
  $out/fuzz run --first=$((91000000001 + w)) --stride=4 --count=100000 --seconds=1200 --results=$out/small_$w.txt > $out/small_$w.log 2>&1 &
done
for w in 0 1; do
  $out/fuzz run --first=$((1000000000001 + w)) --stride=2 --count=100000 --seconds=1200 --results=$out/long_$w.txt > $out/long_$w.log 2>&1 &
  $out/fuzz run --first=$((2000000000001 + w)) --stride=2 --count=100000 --seconds=1200 --results=$out/medium_$w.txt > $out/medium_$w.log 2>&1 &
done
wait
wc -l $out/*.v64
