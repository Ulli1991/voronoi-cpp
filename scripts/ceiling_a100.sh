#!/bin/bash
# The measurements of docs/measure_2026-11-12.md on one A100: sbatch -o build-nvtx/jobs/%x-%j.out scripts/ceiling_a100.sh <mode>
#SBATCH -J ceiling
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
bin=build-nvtx/jobs
exe=$bin/voronoi_bench_nvtx
mode="$1"
shift
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,memory.used --format=csv,noheader -l 5 >"$bin/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed
M+=,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
M+=,lts__t_bytes.sum,l1tex__t_bytes.sum,lts__t_sector_hit_rate.pct,launch__registers_per_thread,launch__block_size
M+=,launch__grid_size,launch__waves_per_multiprocessor,sm__maximum_warps_per_active_cycle_pct
M+=,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active
M+=,smsp__thread_inst_executed_per_inst_executed.ratio,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed
M+=,sm__cycles_elapsed.avg.per_second
for st in long_scoreboard lg_throttle short_scoreboard mio_throttle wait math_pipe_throttle no_instruction branch_resolving \
    barrier membar selected not_selected; do
    M+=",smsp__warp_issue_stalled_${st}_per_warp_active.pct"
done
# ncu <tag> <phase ranges, comma-separated> -- <bench arguments>: every kernel inside the ranges, named by its Kokkos label
ncu_run() {
    local tag="$1" ranges="$2"
    shift 3
    local inc=()
    IFS=',' read -ra rs <<<"$ranges"
    for r in "${rs[@]}"; do inc+=(--nvtx-include "$r/"); done
    dcgmi profile --pause 2>/dev/null
    timeout 820 "$NCU" --target-processes all --clock-control none --nvtx "${inc[@]}" --print-nvtx-rename kernel \
        --metrics "$M" --csv --page raw "$exe" "$@" --nvtx >"$bin/ncu_$tag-$SLURM_JOB_ID.csv"
    echo "ncu $tag exit $?"
    dcgmi profile --resume 2>/dev/null
}
case "$mode" in
smoke)
    ncu_run smoke s2_update,s2_cells -- step 32768 jittered 0.1 2
    head -c 3000 "$bin/ncu_smoke-$SLURM_JOB_ID.csv"
    ;;
step)  # step <d>: kernels of step 3 (update, update_cells), 128³ jittered
    ncu_run "step_$1" s3_update,s3_cells -- step 2097152 jittered "$1" 3
    ;;
start)  # exact start and its update_cells, 128³ jittered
    ncu_run start start,start_cells -- step 2097152 jittered 0.03 0
    ;;
rebuild)  # rebuild <kind>: cell list and warp clipping, one repeat
    ncu_run "rebuild_$1" list,clip -- rebuild 2097152 "$1" 1 --no-check
    ;;
nsys)  # launches, syncs and kernel times per phase: steps at 0.03 and 0.1, rebuilds. [labels]: per-label ranges (fenced)
    module load nsight_systems/2026
    nv=--nvtx=phases; [[ "${1:-}" == labels ]] && nv=--nvtx
    for d in 0.03 0.1; do
        tag="$bin/nsys_step_$d-$SLURM_JOB_ID"
        nsys profile -t cuda,nvtx --cuda-memory-usage=false -s none --cpuctxsw=none -o "$tag" --force-overwrite true \
            "$exe" step 2097152 jittered "$d" 10 $nv | grep -E "^(mesh|gpuflips)"
        nsys export --type sqlite -o "$tag.sqlite" --force-overwrite true "$tag.nsys-rep" >/dev/null 2>&1
    done
    for kind in jittered uniform; do
        tag="$bin/nsys_rebuild_$kind-$SLURM_JOB_ID"
        nsys profile -t cuda,nvtx --cuda-memory-usage=false -s none --cpuctxsw=none -o "$tag" --force-overwrite true \
            "$exe" rebuild 2097152 "$kind" 5 --no-check $nv | grep warp
        nsys export --type sqlite -o "$tag.sqlite" --force-overwrite true "$tag.nsys-rep" >/dev/null 2>&1
    done
    ;;
rate)  # the plain rates of the same binary without --nvtx, for the baseline check
    for d in 0.03 0.1 0.3; do $exe step 2097152 jittered "$d" 10 | grep -E "^(start|mesh|gpuflips)"; done
    for kind in jittered uniform; do $exe rebuild 2097152 "$kind" 5 --no-check | grep warp; done
    ;;
esac
