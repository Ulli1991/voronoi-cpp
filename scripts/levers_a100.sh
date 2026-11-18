#!/bin/bash
# The step levers (branch step-levers) on one A100: builds voronoi_bench of each named tree, then measures them in turn.
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_a100.sh <mode> <name>=<source> [<name>=<source> ...] [-- args]
# A clean tree's binary is cached by commit (build-jobs/levers/cache), so a reference is built once.
# Modes: kernels (per-kernel ms, 128³ jittered, three spacings), rates (128³ and 256³ jittered, three spacings, ten steps,
# two rounds interleaved), long (500-step smooth flow, jittered; args: n, spacings), bits (geometry, faces and lists
# checksums, seven inputs, 0.03 / 0.1 / 0.3, 128³, five steps), ncu (Nsight Compute on step 3 of 128³ jittered at the
# spacings in args (default 0.1), kernels named by Kokkos label through scripts/kp_profile.cpp), memory (--memory phases),
# verify (every step against a fresh build; args: n, spacings, steps), ctest (all targets of the first tree, ctest).
#SBATCH -J levers
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
here=/raven/u/uli/voronoi-cpp-levers
mode="$1"
shift
names=() args=()
while (($#)); do
    if [[ "$1" == "--" ]]; then
        shift
        args=("$@")
        break
    fi
    names+=("${1%%=*}")
    src[${#names[@]}]="${1#*=}"
    shift
done
out="$here/build-jobs/levers"
bin="$out/bin-${SLURM_JOB_ID:-0}"
cache="$out/cache"
mkdir -p "$bin" "$cache"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics,clocks.mem --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 5 >"$bin/clock.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF)
if [[ "$mode" == ctest ]]; then  # every target of the first tree, then ctest
    dir="$bin/build-${names[0]}"
    cmake -S "${src[1]}" -B "$dir" "${cmake_args[@]}" >"$dir.configure.log" 2>&1
    cmake --build "$dir" --parallel 18 2>&1 | grep -E " error|Error " | head -20
    echo "build ${names[0]} ($(git -C "${src[1]}" rev-parse --short HEAD)$(git -C "${src[1]}" diff --quiet || echo +))"
    ctest --test-dir "$dir" --output-on-failure 2>&1 | tail -60
    exit 0
fi
k=0
builds=()
for name in "${names[@]}"; do
    k=$((k + 1))
    dir="$bin/build-$name"
    s="${src[$k]}"
    (
        hash=$(git -C "$s" rev-parse --short HEAD)
        clean=1
        # Only a worktree root with no changes is cached; a snapshot directory inside a tree never is.
        [[ "$(git -C "$s" rev-parse --show-toplevel 2>/dev/null)" == "$(realpath "$s")" ]] || clean=0
        git -C "$s" diff --quiet HEAD -- src CMakeLists.txt || clean=0
        if ((clean)) && [[ -x "$cache/voronoi_bench_$hash" ]]; then
            cp "$cache/voronoi_bench_$hash" "$bin/voronoi_bench_$name"
            echo "build $name ($hash) cached"
            exit 0
        fi
        cmake -S "$s" -B "$dir" "${cmake_args[@]}" -DBUILD_TESTING=OFF -DCMAKE_CXX_FLAGS=-lineinfo >"$dir.configure.log" 2>&1
        cmake --build "$dir" --target voronoi_bench -j6 >"$dir.build.log" 2>&1
        rc=$?
        grep -E " error|warning: .*(spill|stack)" "$dir.build.log" | head -10
        echo "build $name ($hash$( ((clean)) || echo +)) exit $rc"
        cp "$dir/voronoi_bench" "$bin/voronoi_bench_$name"
        ((clean && rc == 0)) && cp "$dir/voronoi_bench" "$cache/voronoi_bench_$hash"
    ) &
    builds+=($!)
done
wait "${builds[@]}"  # not the clock logger
run() {  # run <name> <args...>: one bench run, its lines prefixed by the name and arguments
    local name="$1"
    shift
    "$bin/voronoi_bench_$name" "$@" 2>&1 | grep -E "^(gpuflips|geometry_checksum|mesh|kernel|final_rebuild|layout|verify|memory|refused)" |
        sed "s#^#$name $* | #" | tee -a "$bin/lines.txt"  # '#': the arguments may hold paths
}
measure() {  # measure <mode>: one mode; a job may chain several as kernels+ncu
case "$1" in
kernels)
    for d in "${args[@]:-0.03 0.1 0.3}"; do
        for dd in $d; do
            for name in "${names[@]}"; do run "$name" step 2097152 jittered "$dd" 10 --kernels --no-start; done
        done
    done
    ;;
rates | rates128 | rates256)  # rates128 and rates256: one size only
    sizes=(2097152 16777216)
    [[ "$1" == rates128 ]] && sizes=(2097152)
    [[ "$1" == rates256 ]] && sizes=(16777216)
    for n in "${sizes[@]}"; do
        for d in 0.03 0.1 0.3; do
            for rep in 1 2; do
                for name in "${names[@]}"; do run "$name" step "$n" jittered "$d" 10 --no-start; done
            done
        done
    done
    ;;
long)  # args: <n> <spacings>
    n="${args[0]}"
    for d in ${args[1]}; do
        for name in "${names[@]}"; do run "$name" step "$n" jittered "$d" 500 --no-start; done
    done
    ;;
bits)
    for kind in jittered uniform clustered halo lattice wave cosmicweb; do
        for d in 0.03 0.1 0.3; do
            for name in "${names[@]}"; do run "$name" step 2097152 "$kind" "$d" 5 --checksum --no-start; done
        done
    done
    ;;
memory)
    for name in "${names[@]}"; do
        "$bin/voronoi_bench_$name" step 2097152 jittered 0.1 6 --no-start --memory 2>&1 | grep -E "memory: phase|B/cell" |
            sed "s#^#$name | #"
    done
    ;;
verify)  # args: <n> <spacings> <steps>
    for d in ${args[1]}; do
        for name in "${names[@]}"; do run "$name" step "${args[0]}" jittered "$d" "${args[2]}" --verify --no-start; done
    done
    ;;
ncu)
    module load nsight_systems/2026 >/dev/null 2>&1
    NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    g++ -O2 -shared -fPIC -std=c++17 -I"$CUDA_HOME/include" "$here/scripts/kp_profile.cpp" -o "$bin/libkp_profile.so" -ldl
    M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed
    M+=,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,lts__t_bytes.sum
    M+=,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_dfma_pred_on.sum
    M+=,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum
    M+=,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,launch__registers_per_thread
    M+=,launch__occupancy_limit_registers,launch__block_size,launch__grid_size,smsp__thread_inst_executed_per_inst_executed.ratio
    M+=,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,smsp__inst_executed.sum
    M+=,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg.per_second
    # Where local memory (spills) is served, and warps per scheduler (for the screen's register cap, tried.md).
    M+=,l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct,l1tex__t_sector_pipe_lsu_mem_local_op_st_hit_rate.pct
    M+=,smsp__warps_active.avg.per_cycle_active
    M+=,smsp__warps_eligible.avg.per_cycle_active,l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum
    M+=,l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit.sum
    M+=,l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss.sum
    known=$("$NCU" --query-metrics --chip ga100 2>/dev/null | awk '{print $1}')  # drop names this ncu lacks
    [[ -n "$known" ]] && M=$(tr , "\n" <<<"$M" | while read -r m; do grep -qx "${m%%.*}" <<<"$known" && echo "$m"; done | paste -sd,)
    for st in long_scoreboard short_scoreboard math_pipe_throttle lg_throttle mio_throttle tex_throttle branch_resolving \
        no_instruction barrier wait not_selected selected dispatch_stall drain misc membar imc_miss; do
        M+=",smsp__warp_issue_stalled_${st}_per_warp_active.pct"
    done
    dcgmi profile --pause 2>/dev/null
    for d in "${args[@]:-0.1}"; do
        for name in "${names[@]}"; do
            csv="$bin/ncu_${name}_$d.csv"
            KOKKOS_TOOLS_LIBS="$bin/libkp_profile.so" KP_NVTX=1 timeout 900 "$NCU" --target-processes all --clock-control none \
                --nvtx --print-nvtx-rename kernel --kernel-name-base demangled --metrics "$M" --csv --page raw \
                --nvtx-include "s3_update/" --nvtx-include "s3_cells/" \
                "$bin/voronoi_bench_$name" step 2097152 jittered "$d" 3 --no-start --nvtx=phases >"$csv" 2>"$csv.err"
            echo "## ncu $name $d exit $?"
            python3 "$here/scripts/profile_ncu.py" "$csv" 2097152
        done
    done
    dcgmi profile --resume 2>/dev/null
    ;;
*)
    echo "unknown mode $1"
    exit 2
    ;;
esac
}
IFS=+ read -ra modes <<<"$mode"
for one in "${modes[@]}"; do measure "$one"; done
for one in "${modes[@]}"; do  # tables of the lines above
    [[ "$one" =~ ^(kernels|rates|rates128|rates256|long|bits)$ ]] && python3 "$here/scripts/step2x_tables.py" "${one%[0-9][0-9][0-9]}" "$bin/lines.txt"
done
nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader
