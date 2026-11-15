#!/bin/bash
# kNN-first clipping (branch fewer-clips) against main's walk on one A100, every design in the same job:
#   sbatch --export=ALL,DESIGNS="ref new",LISTS="0 24 32 48" -o build-jobs/%x-%j.out scripts/fewer_clips_a100.sh <mode> [args]
# Binaries: build-jobs/bin/voronoi_bench_<design>; LISTS: VORONOI_KNN values for the `new` design (0: the walk,
# wN: the walk with VORONOI_BALLS=N; "+" for the comma of k,k2, which sbatch --export would split).
#SBATCH -J fewer_clips
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --export=ALL
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
bin=${BIN:-build-jobs/bin}
mode="$1"
shift
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
designs="${DESIGNS:-ref new}"
lists="${LISTS:-0 24 32 48}"
kinds="${KINDS:-jittered uniform clustered halo lattice wave}"
# Every (design, list) pair: ref runs once, new once per list.
pairs() { for d in $designs; do if [[ $d == ref ]]; then echo "ref:0"; else for k in $lists; do echo "$d:$k"; done; fi; done; }
run() {  # run <design:list> <args...>
    local d=${1%%:*} k=${1#*:}
    shift
    if [[ $k == 0 ]]; then env -u VORONOI_KNN -u VORONOI_BALLS $bin/voronoi_bench_$d "$@"
    elif [[ $k == w* ]]; then env -u VORONOI_KNN VORONOI_BALLS=${k#w} $bin/voronoi_bench_$d "$@"
    else env -u VORONOI_BALLS VORONOI_KNN=${k//+/,} $bin/voronoi_bench_$d "$@"; fi
}
case "$mode" in
rate)  # rate <n> <rounds>: construct_ms of every design, interleaved
    for r in $(seq "$2"); do
        for kind in $kinds; do
            for p in $(pairs); do
                echo "== rate $p $kind $1 round $r"
                run $p rebuild "$1" "$kind" 5 --no-check | grep -E "^(warp|listed)"
            done
        done
    done
    ;;
work)  # work <n>: clips, cuts, entries and the cell check of one build
    for kind in $kinds; do
        for p in $(pairs); do
            echo "== work $p $kind $1"
            run $p rebuild "$1" "$kind" 1 | grep -vE "^#"
        done
    done
    ;;
start)  # start <n> <rounds>: the exact start (start_ms), interleaved
    for r in $(seq "$2"); do
        for kind in $kinds; do
            for p in $(pairs); do
                echo "== start $p $kind $1 round $r"
                run $p step "$1" "$kind" 0.1 0 | grep -E "^start"
            done
        done
    done
    ;;
checksum)  # checksum <n>: checksums after the build and two steps
    for kind in $kinds; do
        for p in $(pairs); do
            echo "== checksum $p $kind $1"
            run $p step "$1" "$kind" 0.1 2 --checksum | grep -E "^(start|mesh|geometry_checksum)"
        done
    done
    ;;
measure)  # measure <n>: the measurement build's counters (tested = candidate reads)
    for kind in $kinds; do
        for p in $(pairs); do
            echo "== measure $p $kind $1"
            run $p rebuild "$1" "$kind" 1 --no-check --measure=1000000000 | grep -E "^(listed|measure: (tested|clipped|cuts) )"
        done
    done
    ;;
ncu)  # ncu <n>: warp instructions and time of the clipping kernels and the kNN pass, one build each
    NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    dcgmi profile --pause 2>/dev/null
    for kind in $kinds; do
        for p in $(pairs); do
            echo "== ncu $p $kind $1"
            run $p rebuild "$1" "$kind" 1 --no-check >/dev/null 2>&1  # warm-up outside the profiler
            d=${p%%:*} k=${p#*:}
            [[ $k == 0 ]] && unset VORONOI_KNN VORONOI_BALLS
            [[ $k == w* ]] && { unset VORONOI_KNN; export VORONOI_BALLS=${k#w}; }
            [[ $k != 0 && $k != w* ]] && { unset VORONOI_BALLS; export VORONOI_KNN=${k//+/,}; }
            timeout 600 $NCU --target-processes all --clock-control none --kernel-name-base demangled \
                --kernel-name regex:"warp::Kernel|list_nearest" --launch-count 3 \
                --metrics smsp__inst_executed.sum,gpu__time_duration.sum,launch__registers_per_thread \
                $bin/voronoi_bench_$d rebuild "$1" "$kind" 1 --no-check 2>&1 | grep -E "Kernel|list_nearest|inst_executed|time_duration|registers"
            unset VORONOI_KNN VORONOI_BALLS
        done
    done
    dcgmi profile --resume 2>/dev/null
    ;;
esac
