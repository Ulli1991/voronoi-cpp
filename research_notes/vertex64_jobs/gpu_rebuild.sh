#!/bin/bash
#SBATCH -p gpudev
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH -J v64-gpu
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/gpu_rebuild-%j.out
# #64 on one A100: the six benchmark inputs at 128^3 and the #58 cases, default build (time, checksums) against the
# diagnostic build (time, counters, exact comparison of the flagged cells). Copies of the binaries are run.
module purge; module load gcc/14 cuda/13.0
cd /raven/u/uli/voronoi-cpp-vertex64
out=/ptmp/uli/voronoi-cpp-vertex64/gpu_rebuild_$SLURM_JOB_ID
mkdir -p $out
cp build-cuda/voronoi_bench $out/bench_default
cp build-cuda-v64/voronoi_bench $out/bench_v64
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader
cases="2097152:jittered 2097152:uniform 2097152:halo 2097152:cosmicweb 2097152:fccsolid 2097152:surface
       262144:uniform 16777216:uniform 16777216:halo"
for c in $cases; do
  n=${c%%:*}; kind=${c##*:}
  echo "=== $kind $n default"
  stdbuf -oL $out/bench_default rebuild $n $kind 5 2>&1 | tee $out/default_${kind}_$n.txt | grep -E "^warp:|clip_checksum|faces|PASS|FAIL"
  echo "=== $kind $n v64"
  VERTEX64_DUMP=$out/cells_${kind}_$n.csv stdbuf -oL $out/bench_v64 rebuild $n $kind 5 2>&1 | tee $out/v64_${kind}_$n.txt \
    | grep -E "^warp:|clip_checksum|faces|PASS|FAIL|^v64"
done
