#!/bin/bash
#SBATCH -p gpudev
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=200G
#SBATCH --time=00:15:00
#SBATCH -J v64-fzg
#SBATCH -o /raven/u/uli/voronoi-cpp-joblogs/vertex64/fuzz_gpu-%j.out
# #64: GPU fuzz (CUDA diagnostic build, warp kernel), counters per seed in RESULTS.v64.
# Profiles medium, large, huge, giant on one A100 each, 12 minutes.
module purge; module load gcc/14 cuda/13.0
cd /raven/u/uli/voronoi-cpp-vertex64
out=/ptmp/uli/voronoi-cpp-vertex64/fuzz_gpu_$SLURM_JOB_ID
mkdir -p $out
cp build-cuda-v64/mesh_fuzz_test $out/fuzz
g=0
for p in 2 3 4 7; do
  CUDA_VISIBLE_DEVICES=$g $out/fuzz run --first=$((p * 1000000000000 + 64000001)) --count=100000 --seconds=720 \
    --results=$out/profile_$p.txt > $out/profile_$p.log 2>&1 &
  g=$((g + 1))
done
wait
wc -l $out/*.v64
