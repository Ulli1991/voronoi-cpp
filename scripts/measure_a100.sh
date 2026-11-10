#!/bin/bash
# The measurements of docs/measure_2026-11-10.md on one A100: sbatch -o build-measure/jobs/%x-%j.out scripts/measure_a100.sh <mode>
#SBATCH -J measure
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
bin=build-measure/jobs
mode="$1"
shift
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,memory.used --format=csv,noheader -l 5 >"$bin/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
inputs() {  # n kind box for one size: 128 or 256
    case "$1" in
    128) echo "2097152 jittered 1;2097152 uniform 1;2097152 clustered 1;2097152 halo 1;4194304 jittered 2,2,0.0625" ;;
    256) echo "16777216 jittered 1;16777216 uniform 1;16777216 clustered 1;16777216 halo 1;16777216 jittered 2,2,0.03125" ;;
    esac
}
case "$mode" in
smoke)
    $bin/voronoi_bench_measure rebuild 262144 clustered warp 1 --no-check --measure=64 --dump=$bin/smoke.npy
    $bin/voronoi_bench_measure step 262144 jittered 0.1 3 --measure
    ;;
clip)  # clip <128|256> [every]: counters per cell, cycle tails, shuffle test
    IFS=';' read -ra rows <<<"$(inputs "$1")"
    for row in "${rows[@]}"; do
        read -r n kind box <<<"$row"
        tag="${kind}_$n"; [[ "$box" != 1 ]] && tag="slab_$n"
        echo "== clip $tag box=$box"
        dump=""; [[ "$1" == 128 ]] && dump="--dump=$bin/clip_$tag.npy"
        $bin/voronoi_bench_measure rebuild "$n" "$kind" warp 1 --no-check --box="$box" --measure="${2:-1024}" $dump
    done
    ;;
rate)  # rate <128|256>: plain rebuild rates of the same inputs, with kernel times
    IFS=';' read -ra rows <<<"$(inputs "$1")"
    for row in "${rows[@]}"; do
        read -r n kind box <<<"$row"
        echo "== rate $kind $n box=$box"
        $bin/voronoi_bench_plain rebuild "$n" "$kind" warp 5 --no-check --box="$box"
    done
    ;;
step)  # step <n> <kind> <box> <measured steps> <displacements...>: changes per step (measure build), kernel times (plain)
    n="$1" kind="$2" box="$3" measured="$4"; shift 4
    for d in "$@"; do
        echo "== step $kind $n box=$box d=$d"
        $bin/voronoi_bench_plain step "$n" "$kind" "$d" 6 --box="$box" --kernels | grep -E "^(gpuflips|kernel|mesh|start)"
        $bin/voronoi_bench_measure step "$n" "$kind" "$d" "$measured" --box="$box" --measure | grep -E "^(step_measure|gpuflips)"
    done
    ;;
ncu)  # ncu <rebuild|step> <n> <kind> [box]: registers, local memory, occupancy, warp stalls per kernel
    NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    what="$1" n="$2" kind="$3" box="${4:-1}"
    HW=gpu__time_duration.sum,launch__registers_per_thread,launch__block_size,launch__grid_size,launch__occupancy_limit_registers
    HW+=,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,sm__warps_active.avg.pct_of_peak_sustained_active
    HW+=,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum
    HW+=,sm__sass_inst_executed_op_local_ld.sum,sm__sass_inst_executed_op_local_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
    HW+=,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed
    HW+=,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct
    for st in long_scoreboard short_scoreboard math_pipe_throttle lg_throttle mio_throttle tex_throttle branch_resolving no_instruction \
        barrier wait sleeping not_selected selected dispatch_stall drain misc membar imc_miss; do
        HW+=",smsp__warp_issue_stalled_${st}_per_warp_active.pct"
    done
    common=(--target-processes all --clock-control none --print-kernel-base demangled --kernel-name-base demangled --metrics "$HW" --csv --page raw)
    dcgmi profile --pause 2>/dev/null
    if [[ "$what" == rebuild ]]; then
        timeout 800 "$NCU" "${common[@]}" --kernel-name "regex:warp::Kernel" --launch-count 1 \
            $bin/voronoi_bench_plain rebuild "$n" "$kind" warp 1 --no-check --box="$box" >"$bin/ncu_rebuild_${kind}_$n-$SLURM_JOB_ID.csv"
    else
        re='(snap|move_screened|circumcentres|number_faces|update_face_rings|tet_mesh::faces|cells_incremental|finish_incremental|CellSortWarp|CellSumsWarp|tet_repair::repair|total_volume|sort_and_sum)'
        timeout 800 "$NCU" "${common[@]}" --kernel-name "regex:$re" --launch-count 400 \
            $bin/voronoi_bench_plain step "$n" "$kind" "${5:-0.1}" 3 --box="$box" >"$bin/ncu_step_${kind}_${n}_${5:-0.1}-$SLURM_JOB_ID.csv"
    fi
    echo "ncu exit $?"
    dcgmi profile --resume 2>/dev/null
    ;;
nsys)  # nsys <n> <kind> <d>: launches, copies and gaps of 20 steps against 10 (difference = 10 steps)
    module load nsight_systems/2026
    for steps in 10 20; do
        tag="$bin/nsys_${2}_$1_$3_$steps-$SLURM_JOB_ID"
        nsys profile -t cuda --cuda-memory-usage=false -s none --cpuctxsw=none -o "$tag" --force-overwrite true \
            $bin/voronoi_bench_plain step "$1" "$2" "$3" "$steps" | grep -c gpuflips
        nsys stats -q --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv -o "$tag" "$tag.nsys-rep" >/dev/null 2>&1
        nsys export --type sqlite -o "$tag.sqlite" --force-overwrite true "$tag.nsys-rep" >/dev/null 2>&1
    done
    ;;
memory)  # memory <n> <kind>: arrays at the peak of a moving step
    $bin/voronoi_bench_plain step "$1" "$2" 0.1 4 --memory
    ;;
esac
