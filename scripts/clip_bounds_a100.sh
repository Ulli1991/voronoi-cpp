#!/bin/bash
# P1 and R8 (docs/ideas.md) on one A100: sbatch -o jobs/%x-%j.out scripts/clip_bounds_a100.sh <mode> [args]
# Binaries in jobs/: voronoi_bench_main (origin/main), voronoi_bench_new (this branch), voronoi_bench_measure.
#SBATCH -J clip_bounds
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
bin=jobs
new="${NEW:-new}"  # tag of this branch's binary, jobs/voronoi_bench_$new
mode="$1"
shift
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 5 >"$bin/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
case "$mode" in
bound)  # bound <n>: counters per cell of the unchanged walk, with the P1 and R8 tests evaluated on the side
    for kind in jittered uniform clustered halo; do
        echo "== bound $kind $1"
        $bin/voronoi_bench_measure rebuild "$1" "$kind" 1 --no-check --measure=1024 | grep -v shuffle
    done
    ;;
counts)  # counts <n>: the same counters with the bounds in use (measurement build of this branch)
    for kind in jittered uniform clustered halo; do
        echo "== counts $kind $1"
        $bin/voronoi_bench_measure_new rebuild "$1" "$kind" 1 --no-check --measure=1024 | grep -v shuffle
    done
    ;;
rate)  # rate <n> <rounds>: rebuild rates, main and this branch interleaved
    for r in $(seq "$2"); do
        for kind in jittered uniform clustered halo lattice wave; do
            for b in main "$new"; do
                echo "== rate $b $kind $1 round $r"
                $bin/voronoi_bench_$b rebuild "$1" "$kind" 5 --no-check | grep -E "^warp"
            done
        done
    done
    ;;
step)  # step <n> <rounds> <displacements...>: exact start and moving steps, main and this branch interleaved
    n="$1" rounds="$2"
    shift 2
    for r in $(seq "$rounds"); do
        for d in "$@"; do
            for b in main "$new"; do
                echo "== step $b jittered $n d=$d round $r"
                $bin/voronoi_bench_$b step "$n" jittered "$d" 10 | grep -E "^(start|mesh|gpuflips)"
            done
        done
    done
    ;;
checksum)  # checksum <n>: cells_checksum after the build and geometry checksums of two steps, main and this branch
    for kind in jittered uniform clustered halo lattice wave; do
        for b in main "$new"; do
            echo "== checksum $b $kind $1"
            $bin/voronoi_bench_$b step "$1" "$kind" 0.1 2 --checksum | grep -E "^(start|mesh|geometry_checksum)"
        done
    done
    ;;
ncu)  # ncu <n> <kinds...>: instructions, local memory, occupancy and stalls of the warp kernel, main and this branch
    NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    n="$1"
    shift
    HW=gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active
    HW+=,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__inst_executed.sum
    HW+=,sm__sass_inst_executed_op_local_ld.sum,sm__sass_inst_executed_op_local_st.sum,sm__sass_inst_executed_op_shared_ld.sum
    for st in long_scoreboard short_scoreboard math_pipe_throttle lg_throttle mio_throttle branch_resolving no_instruction wait not_selected selected; do
        HW+=",smsp__warp_issue_stalled_${st}_per_warp_active.pct"
    done
    dcgmi profile --pause 2>/dev/null
    for kind in "$@"; do
        for b in main "$new"; do
            echo "== ncu $b $kind $n"
            timeout 400 "$NCU" --target-processes all --clock-control none --print-kernel-base demangled --kernel-name-base demangled --kernel-name "regex:warp::Kernel" --launch-count 1 \
                --metrics "$HW" --csv --page raw $bin/voronoi_bench_$b rebuild "$n" "$kind" 1 --no-check >"$bin/ncu_${b}_$kind-$SLURM_JOB_ID.csv" 2>&1
            echo "ncu exit $?"
        done
    done
    dcgmi profile --resume 2>/dev/null
    ;;
esac
