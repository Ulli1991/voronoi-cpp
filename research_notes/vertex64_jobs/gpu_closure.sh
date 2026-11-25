#!/bin/bash
#SBATCH -p gpudev
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH -J v64-gpu4
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/gpu_closure-%j.out
# #64 on one A100: rebuild the diagnostic, then every cell of the plain warp rebuild against exact cells
# (VERTEX64_ALL) on the inputs that flag, with the default build's check for reference.
module purge; module load gcc/14 cuda/13.0 cmake
cd /raven/u/uli/voronoi-cpp-vertex64
cmake --build build-cuda-v64 -j18 --target voronoi_bench 2>&1 | grep -E " error|Built"
out=/ptmp/uli/voronoi-cpp-vertex64/gpu_closure_$SLURM_JOB_ID
mkdir -p $out
cp build-cuda-v64/voronoi_bench $out/bench_v64
for c in 262144:sc@1e-9 262144:fcc@1e-9 262144:shells@1e-9 262144:bcc@0 262144:fcc@0 262144:wave 262144:uniform 2097152:halo 2097152:surface; do
  n=${c%%:*}; kind=${c##*:}
  echo "=== $kind $n v64 all"
  VERTEX64_ALL=1 VERTEX64_DUMP=$out/cells_${kind}_$n.csv stdbuf -oL $out/bench_v64 rebuild $n $kind 1 2>&1 \
    | tee $out/v64_${kind}_$n.txt | grep -E "^warp:|clip_checksum|faces|PASS|FAIL|^v64"
done
