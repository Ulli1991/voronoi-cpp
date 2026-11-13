#!/bin/bash
# Canonical rings and the lean cut (branch clip-canonical) against main on one A100, every design in the same job:
#   sbatch --export=ALL,DESIGNS="ref canon",KINDS=uniform -o build-jobs/%x-%j.out scripts/clip_canonical_a100.sh <mode> [args]
# Binaries: build-jobs/bin/voronoi_bench_<design> (ref: main with the bench's checksums).
#SBATCH -J clip_canon
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
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 5 >"build-jobs/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
designs="${DESIGNS:-ref canon}"
kinds="${KINDS:-jittered uniform clustered halo lattice wave}"
case "$mode" in
rate)  # rate <n> <rounds>: construct_ms of every design, interleaved
    for r in $(seq "$2"); do
        for kind in $kinds; do
            for d in $designs; do
                echo "== rate $d $kind $1 round $r"
                $bin/voronoi_bench_$d rebuild "$1" "$kind" 5 --no-check | grep -E "^warp"
            done
        done
    done
    ;;
start)  # start <n> <rounds>: the exact start (start_ms), interleaved
    for r in $(seq "$2"); do
        for kind in $kinds; do
            for d in $designs; do
                echo "== start $d $kind $1 round $r"
                $bin/voronoi_bench_$d step "$1" "$kind" 0.1 0 | grep -E "^start"
            done
        done
    done
    ;;
checksum)  # checksum <n> [check]: checksums after the build and two steps (and with check: the rebuild's cell check)
    for kind in $kinds; do
        for d in $designs; do
            echo "== checksum $d $kind $1"
            $bin/voronoi_bench_$d step "$1" "$kind" 0.1 2 --checksum | grep -E "^(start|mesh|geometry_checksum)"
            [[ "${2:-}" == check ]] && $bin/voronoi_bench_$d rebuild "$1" "$kind" 1 | grep -vE "^(#|warp)"
        done
    done
    ;;
ncu)  # ncu <n> [section]: the warp kernel's instructions, stalls and source counters (-lineinfo binaries)
    NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    dcgmi profile --pause 2>/dev/null
    for kind in $kinds; do
        for d in $designs; do
            echo "== ncu $d $kind $1"
            timeout 800 $NCU --target-processes all --clock-control none --kernel-name-base demangled \
                --kernel-name regex:warp::Kernel --launch-count 1 --section ${2:-SourceCounters} --section SpeedOfLight \
                --section LaunchStats --section WarpStateStats --import-source yes \
                -f -o build-jobs/ncu_${d}_$kind-$SLURM_JOB_ID $bin/voronoi_bench_$d rebuild "$1" "$kind" 1 --no-check \
                >/dev/null 2>&1
            echo "ncu exit $?"
        done
    done
    dcgmi profile --resume 2>/dev/null
    ;;
esac
