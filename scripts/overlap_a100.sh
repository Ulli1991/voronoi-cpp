#!/bin/bash
# GPU busy and idle time, host waits and tail work of the warp rebuild, the exact start and moving steps on one A100
# (docs/measurements.md, "CPU-GPU overlap"). From the repository root, with build-overlap (plain, -lineinfo) and
# build-overlap-measure (-DVORONOI_MEASURE) built:
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/overlap_a100.sh <128|256> [kinds...]
# Tables: python3 scripts/overlap_tables.py build-overlap/jobs/<job>.
#SBATCH -J overlap
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake nsight_systems/2026
side="$1"
shift
kinds=("${@:-jittered uniform clustered halo cosmicweb surface}")
read -ra kinds <<<"${kinds[*]}"
out="build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out"
# Copies, so a rebuild meanwhile does not change the binaries under the job.
cp build-overlap/voronoi_bench "$out/bench_plain"
cp build-overlap-measure/voronoi_bench "$out/bench_measure"
hostname
lscpu | grep -E "^Model name|^CPU\(s\)"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,memory.used --format=csv,noheader -l 5 >"$out/clock.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
n=$((side * side * side))
export OMP_PROC_BIND=spread OMP_PLACES=cores
profile() {  # profile <tag> <args...>: nsys trace of CUDA and NVTX, exported to sqlite
    local tag="$1"
    shift
    nsys profile -t cuda,nvtx --cuda-memory-usage=false -s none --cpuctxsw=none -o "$out/$tag" --force-overwrite true \
        "$out/bench_plain" "$@" >"$out/$tag.log" 2>&1
    echo "nsys $tag exit $?"
    nsys export --type sqlite -o "$out/$tag.sqlite" --force-overwrite true "$out/$tag.nsys-rep" >/dev/null 2>&1
    rm -f "$out/$tag.nsys-rep"
}
for kind in "${kinds[@]}"; do
    [[ "$kind" == surface && "$side" != 128 ]] && continue
    echo "== $kind $side"
    # Warp rebuild: 3 timed clippings in NVTX ranges; tail counts from the measurement build.
    profile "rebuild_${kind}_$side" rebuild "$n" "$kind" 3 --no-check --nvtx=phases
    grep -E "^warp:" "$out/rebuild_${kind}_$side.log"
    timeout 900 "$out/bench_measure" rebuild "$n" "$kind" 1 --no-check --measure=65536 >"$out/measure_${kind}_$side.log" 2>&1
    grep -E "warp_status|cycle_share" "$out/measure_${kind}_$side.log"
    # Exact start (range "start") and moving steps (ranges s<k>_update, s<k>_cells).
    steps=8
    [[ "$kind" == surface ]] && steps=4
    for d in 0.03 0.1 0.3; do
        profile "step_${kind}_${side}_$d" step "$n" "$kind" "$d" "$steps" --nvtx=phases
        grep -E "^(start|mesh|gpuflips)" "$out/step_${kind}_${side}_$d.log" | cut -c1-400
    done
done
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader
