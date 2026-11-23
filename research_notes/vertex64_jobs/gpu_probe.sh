#!/bin/bash
#SBATCH -p gpudev
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH -J v64-gpu
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/gpu_probe-%j.out
# #64: where the diagnostic build stalls, at small n, line-buffered.
module purge; module load gcc/14 cuda/13.0
out=/ptmp/uli/voronoi-cpp-vertex64/probe_$SLURM_JOB_ID; mkdir -p $out; cp /raven/u/uli/voronoi-cpp-vertex64/build-cuda-v64/voronoi_bench $out/bench_v64
for n in 4096 32768 262144; do
  echo "=== $n"
  timeout 120 stdbuf -oL $out/bench_v64 rebuild $n jittered 1 2>&1 | tail -20
  echo "exit $?"
done
