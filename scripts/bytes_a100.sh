#!/bin/bash
# DRAM bytes per kernel of the moving step for one or more builds (docs/measurements.md, step bytes): Nsight Compute on
# step 3 (update and update_cells), 128³ jittered, at 0.03 and 0.1 spacings.
#   sbatch -o build-cuda/jobs/%x-%j.out scripts/bytes_a100.sh <dir> <tag> [<tag> ...]
# <dir>/voronoi_bench_<tag> are the binaries; tables: scripts/ceiling_tables.py reads the csv files.
#SBATCH -J step_bytes
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
dir="$1"
shift
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread
M+=,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio
M+=,sm__cycles_elapsed.avg.per_second,lts__t_bytes.sum
dcgmi profile --pause 2>/dev/null
for tag in "$@"; do
    for d in 0.03 0.1; do
        timeout 400 "$NCU" --target-processes all --clock-control none --nvtx --nvtx-include "s3_update/" \
            --nvtx-include "s3_cells/" --print-nvtx-rename kernel --metrics "$M" --csv --page raw \
            "$dir/voronoi_bench_$tag" step 2097152 jittered "$d" 3 --nvtx >"$dir/ncu_${tag}_$d-$SLURM_JOB_ID.csv"
        echo "ncu $tag $d exit $?"
    done
done
dcgmi profile --resume 2>/dev/null
