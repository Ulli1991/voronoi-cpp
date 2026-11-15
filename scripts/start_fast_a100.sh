#!/bin/bash
# The exact start of main (ref) against branch start-fast (new) on one A100, both in the same job:
#   sbatch -o build-jobs/%x-%j.out scripts/start_fast_a100.sh <mode> <n> [rounds]   (proof: everything, --time=06:00:00)
# Binaries: build-jobs/bin/voronoi_bench_{ref,new}; KINDS: inputs (bluenoise reads the libbench .pts file).
#SBATCH -J start_fast
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
bin=${BIN:-build-jobs/bin}
designs="${DESIGNS:-ref new}"
kinds="${KINDS:-jittered uniform clustered halo lattice wave bluenoise}"
mode="$1"
n="${2:-0}"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
input() {  # input <kind>: the bench's distribution argument
    if [[ $1 == bluenoise ]]; then echo /raven/u/uli/libbench-data/points/bluenoise_$n.pts; else echo "$1"; fi
}
case "$mode" in
proof)  # proof: the start (3 rounds), its kernels and the checksums at 128³ and 256³, all in this job
    for size in 2097152 16777216; do
        for m in "start $size 3" "kernels $size" "checksum $size"; do
            SLURM_JOB_ID=${SLURM_JOB_ID:-0} bash "$0" $m
        done
    done
    ;;
start)  # start <n> <rounds>: start_ms (clipping + tet_start::build, median of 3), interleaved
    for r in $(seq "${3:-1}"); do
        for kind in $kinds; do
            for d in $designs; do
                echo "== start $d $kind $n round $r"
                $bin/voronoi_bench_$d step "$n" "$(input $kind)" 0.1 0 | grep -E "^(start|mesh)"
            done
        done
    done
    ;;
kernels)  # kernels <n>: the start's kernels, fenced (Kokkos Tools callbacks)
    for kind in $kinds; do
        for d in $designs; do
            echo "== kernels $d $kind $n"
            $bin/voronoi_bench_$d step "$n" "$(input $kind)" 0.1 0 --kernels | grep -E "^(start|mesh)"
        done
    done
    ;;
checksum)  # checksum <n>: the start line, the build's cells_checksum, and the checksums of two steps
    for kind in $kinds; do
        for d in $designs; do
            echo "== checksum $d $kind $n"
            $bin/voronoi_bench_$d step "$n" "$(input $kind)" 0.1 2 --checksum | grep -E "^(start|mesh|geometry_checksum)"
        done
    done
    ;;
memory)  # memory <n>: device memory peaks per phase (the start block, then Mesh::update's start)
    for kind in $kinds; do
        for d in $designs; do
            echo "== memory $d $kind $n"
            $bin/voronoi_bench_$d step "$n" "$(input $kind)" 0.1 0 --memory | grep -iE "^(start:|mesh:|memory|phase|device)"
        done
    done
    ;;
nsys)  # nsys <n>: launches, synchronisations and kernel times of the start (one step-bench start block)
    module load nsight_systems/2026
    for kind in $kinds; do
        for d in $designs; do
            echo "== nsys $d $kind $n"
            out=build-jobs/nsys-$SLURM_JOB_ID-$d-$kind-$n
            nsys profile -t cuda,nvtx -o "$out" --force-overwrite true $bin/voronoi_bench_$d step "$n" "$(input $kind)" \
                0.1 0 --nvtx=phases >/dev/null 2>&1
            nsys stats -q -r cuda_gpu_kern_sum,cuda_api_sum -f csv "$out.nsys-rep" 2>/dev/null | head -60
        done
    done
    ;;
ncu)  # ncu <n>: every kernel of one start (the range start_once), named by its Kokkos label
    NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    dcgmi profile --pause 2>/dev/null
    for kind in $kinds; do
        for d in $designs; do
            echo "== ncu $d $kind $n"
            timeout 1200 $NCU --clock-control none --nvtx --nvtx-include "start_once/" --section SpeedOfLight \
                --section Occupancy --section LaunchStats --section WarpStateStats --section MemoryWorkloadAnalysis \
                --metrics smsp__thread_inst_executed_per_inst_executed.ratio,dram__bytes_read.sum,dram__bytes_write.sum \
                --print-nvtx-rename kernel --csv --page raw $bin/voronoi_bench_$d step "$n" "$(input $kind)" 0.1 0 --nvtx \
                >build-jobs/ncu-$SLURM_JOB_ID-$d-$kind-$n.csv 2>&1
            echo "rows $(wc -l <build-jobs/ncu-$SLURM_JOB_ID-$d-$kind-$n.csv)"
        done
    done
    dcgmi profile --resume 2>/dev/null
    ;;
esac
