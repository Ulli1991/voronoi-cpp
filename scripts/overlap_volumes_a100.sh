#!/bin/bash
# The volumes-only path (library_bench build --path=warp-host-vc: points in host memory to volumes and centroids in host
# memory) under Nsight Systems: copies, allocations and kernels per build (docs/measurements.md, "CPU-GPU overlap").
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/overlap_volumes_a100.sh
# Tables: python3 scripts/overlap_tables.py --volumes build-overlap/jobs/<job>
#SBATCH -J overlap_vol
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:45:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake nsight_systems/2026
out="build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out"
cp build-overlap/library_bench "$out/library_bench"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,memory.used --format=csv,noheader -l 5 >"$out/clock.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
for side in 128 256; do
    n=$((side * side * side))
    for kind in jittered uniform clustered halo cosmicweb; do
        pts="$out/points_${kind}_$side.bin"
        "$out/library_bench" points "$n" "$kind" "$pts" >/dev/null
        tag="volumes_${kind}_$side"
        nsys profile -t cuda --cuda-memory-usage=false -s none --cpuctxsw=none -o "$out/$tag" --force-overwrite true \
            "$out/library_bench" build "$pts" --path=warp-host-vc --repeats=3 >"$out/$tag.log" 2>&1
        echo "nsys $tag exit $?"
        nsys export --type sqlite -o "$out/$tag.sqlite" --force-overwrite true "$out/$tag.nsys-rep" >/dev/null 2>&1
        rm -f "$out/$tag.nsys-rep" "$pts"
        grep -iE "ms|cells" "$out/$tag.log" | head -3
    done
done
