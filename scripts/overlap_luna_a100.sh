#!/bin/bash
# luna-cpp moving Sedov 128³ (grad32, the 3D default) under Nsight Systems: GPU idle and host waits per hydro step, and the
# snapshot write (docs/measurements.md, "CPU-GPU overlap"). From the voronoi-cpp repository root:
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/overlap_luna_a100.sh <luna-cpp build dir>
# Tables: python3 scripts/overlap_tables.py --luna build-overlap/jobs/<job>/luna_sedov128.sqlite
#SBATCH -J overlap_luna
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial nsight_systems/2026
build="$(realpath "$1")"
out="build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out/snap" "$out/snap_plain"
cp "$build/hydro_sedov" "$out/hydro_sedov"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,memory.used --format=csv,noheader -l 5 >"$out/clock.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
run=("$out/hydro_sedov" 128 jittered --moving --precision=grad32 --steps=12)
# Without the profiler: the driver's own phase times and the snapshot write, timed from outside.
start=$(date +%s.%N)
"${run[@]}" --out="$out/snap_plain" | grep -vE "^  "
echo "wall_s_with_output=$(echo "$(date +%s.%N) - $start" | bc)"
start=$(date +%s.%N)
"${run[@]}" | grep -E "^(run|phases|throughput|ours3d|mesh)"
echo "wall_s_without_output=$(echo "$(date +%s.%N) - $start" | bc)"
nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=false -s none --cpuctxsw=none -o "$out/luna_sedov128" --force-overwrite true \
    "${run[@]}" --out="$out/snap" >"$out/luna_sedov128.log" 2>&1
echo "nsys exit $?"
nsys export --type sqlite -o "$out/luna_sedov128.sqlite" --force-overwrite true "$out/luna_sedov128.nsys-rep" >/dev/null 2>&1
ls -la "$out/snap" "$out/snap_plain"
