#!/bin/bash
# A lever of the geometry derivatives on one A100 (gpudev): CUDA build of the test and the bench, the test (finite
# differences, dot test, identities), the bench at 128³ (3D jittered, white noise) and 2²⁰ (2D), Nsight Compute rows of
# the tangent and the adjoint at 128³ jittered (kernels by Kokkos label, scripts/profile_ncu.py).
#   sbatch -o <log dir>/%x-%j.out scripts/geomderiv_levers_a100.sh <build dir>
#SBATCH -J gd_lever
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
here=$(pwd)
dir="$1"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics --format=csv,noheader
echo "== source $(git log --oneline -1 | cut -c1-60)$(git diff --quiet HEAD -- src || echo ' +changes')"
cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda -DCMAKE_CXX_FLAGS=-lineinfo \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >/dev/null
cmake --build "$dir" --parallel 18 --target geometry_derivative_test geometry_derivative_bench 2>&1 |
    grep -E " error|error:|warning #|warning:" | head -20
echo "== test"
"$dir/geometry_derivative_test" | grep -E "FAIL|extrapolated|dot test all|passed|FAILED"
echo "== bench"
for args in "3 2097152 jittered 7" "3 2097152 uniform 7" "2 1048576 jittered 7"; do
    "$dir/geometry_derivative_bench" $args
done
echo "== ncu"
NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
g++ -O2 -shared -fPIC -std=c++17 -I"$CUDA_HOME/include" "$here/scripts/kp_profile.cpp" -o "$dir/libkp_profile.so" -ldl
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed
M+=,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,lts__t_bytes.sum
M+=,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_dfma_pred_on.sum
M+=,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum
M+=,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,launch__registers_per_thread
M+=,launch__occupancy_limit_registers,launch__block_size,launch__grid_size,smsp__thread_inst_executed_per_inst_executed.ratio
M+=,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,smsp__inst_executed.sum
M+=,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg.per_second
for st in long_scoreboard short_scoreboard math_pipe_throttle lg_throttle mio_throttle tex_throttle branch_resolving \
    no_instruction barrier wait not_selected selected dispatch_stall drain misc membar imc_miss; do
    M+=",smsp__warp_issue_stalled_${st}_per_warp_active.pct"
done
dcgmi profile --pause 2>/dev/null
csv="$dir/ncu_gd-$SLURM_JOB_ID.csv"
KOKKOS_TOOLS_LIBS="$dir/libkp_profile.so" KP_NVTX=1 timeout 400 "$NCU" --target-processes all --clock-control none \
    --nvtx --print-nvtx-rename kernel --kernel-name-base demangled --metrics "$M" --csv --page raw \
    --nvtx-include "gd_tangent/" --nvtx-include "gd_adjoint/" \
    "$dir/geometry_derivative_bench" 3 2097152 jittered 1 >"$csv" 2>"$csv.err"
echo "## ncu exit $?"
dcgmi profile --resume 2>/dev/null
python3 scripts/profile_ncu.py "$csv" 2097152
