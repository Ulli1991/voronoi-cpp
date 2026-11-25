#!/bin/bash
# A geometry-derivative lever against a reference build in one job pair: `build` (CPU node: CUDA builds of the reference's
# bench and this tree's test and bench, -lineinfo), then `run` (A100, gpudev: the test; both benches at 128³ 3D jittered
# and white noise, 2²⁰ 2D; Nsight Compute rows per kernel of both, scripts/profile_ncu.py).
#   sbatch -p interactive -c 8 --time=01:00:00 scripts/geomderiv_pair_a100.sh build <dir> <ref tree>
#   sbatch -p gpudev --gres=gpu:a100:1 -c 18 --time=00:15:00 scripts/geomderiv_pair_a100.sh run <dir>
#SBATCH -J gd_pair
#SBATCH --ntasks=1
#SBATCH --mem=60G
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
here=$(pwd)
mode="$1"
dir="$2"
configure() {  # configure <source> <build>
    cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda -DCMAKE_CXX_FLAGS=-lineinfo \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >/dev/null
}
if [[ $mode == build ]]; then
    ref="$3"
    echo "== ref $ref, new $(git log --oneline -1 | cut -c1-60)$(git diff --quiet HEAD -- src || echo ' +changes')"
    configure "$ref" "$dir/ref"
    configure . "$dir/new"
    cmake --build "$dir/ref" --parallel "${SLURM_CPUS_PER_TASK:-8}" --target geometry_derivative_bench 2>&1 |
        grep -E " error|error:" | head
    cmake --build "$dir/new" --parallel "${SLURM_CPUS_PER_TASK:-8}" --target geometry_derivative_test \
        geometry_derivative_bench 2>&1 | grep -E " error|error:|warning:" | head -20
    g++ -O2 -shared -fPIC -std=c++17 -I"$CUDA_HOME/include" scripts/kp_profile.cpp -o "$dir/libkp_profile.so" -ldl
    ls -la "$dir/ref/geometry_derivative_bench" "$dir/new/geometry_derivative_bench" "$dir/new/geometry_derivative_test"
    exit 0
fi
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics --format=csv,noheader
echo "== test (new)"
"$dir/new/geometry_derivative_test" | grep -E "FAIL|extrapolated|dot test all|passed|FAILED"
echo "== bench"
for args in "3 2097152 jittered 7" "3 2097152 uniform 7" "2 1048576 jittered 7"; do
    for side in ref new; do echo -n "$side "; "$dir/$side/geometry_derivative_bench" $args; done
done
echo "== ncu"
NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
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
for side in ref new; do
    csv="$dir/ncu_${side}-$SLURM_JOB_ID.csv"
    KOKKOS_TOOLS_LIBS="$dir/libkp_profile.so" KP_NVTX=1 timeout 300 "$NCU" --target-processes all --clock-control none \
        --nvtx --print-nvtx-rename kernel --kernel-name-base demangled --metrics "$M" --csv --page raw \
        --nvtx-include "gd_tangent/" --nvtx-include "gd_adjoint/" \
        "$dir/$side/geometry_derivative_bench" 3 2097152 jittered 1 >"$csv" 2>"$csv.err"
    echo "## ncu $side exit $?"
    python3 scripts/profile_ncu.py "$csv" 2097152
done
dcgmi profile --resume 2>/dev/null
