#!/bin/bash
#SBATCH -p gpudev
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH -J v64-gpu2
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/gpu_degenerate-%j.out
# #64 on one A100: degenerate inputs (lattices, cospherical shells, planes, pairs) where the fuzz flags fire;
# diagnostic build (time, counters, exact comparison of the flagged cells). Copies of the binaries are run.
module purge; module load gcc/14 cuda/13.0
cd /raven/u/uli/voronoi-cpp-vertex64
out=/ptmp/uli/voronoi-cpp-vertex64/gpu_degenerate_$SLURM_JOB_ID
mkdir -p $out
cp build-cuda/voronoi_bench $out/bench_default
cp build-cuda-v64/voronoi_bench $out/bench_v64
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader
cases="262144:lattice 262144:wave 262144:sc@0 262144:bcc@0 262144:fcc@0 262144:sc@1e-9 262144:fcc@1e-9 262144:shells@1e-9 262144:planes@0 262144:pairs@1e-9 2097152:clustered
"
for c in $cases; do
  n=${c%%:*}; kind=${c##*:}
  echo "=== $kind $n default"
  stdbuf -oL $out/bench_default rebuild $n $kind 5 2>&1 | tee $out/default_${kind}_$n.txt | grep -E "^warp:|clip_checksum|faces|PASS|FAIL"
  echo "=== $kind $n v64"
  VERTEX64_DUMP=$out/cells_${kind}_$n.csv stdbuf -oL $out/bench_v64 rebuild $n $kind 5 2>&1 | tee $out/v64_${kind}_$n.txt \
    | grep -E "^warp:|clip_checksum|faces|PASS|FAIL|^v64"
done
