#!/bin/bash
# S0 and S2 of "CPU and GPU together" (docs/ideas.md) on one gpu1 node, from the repository root:
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/modes_a100.sh
# 1. Build cost: voronoi_bench against the CUDA-only Kokkos (the build of main) and the CUDA + OpenMP Kokkos
#    (/u/uli/kokkos-cuda-omp, scripts/kokkos_cuda_omp.sh), side by side, one job each (-j1): seconds and bytes.
# 2. ctest of both builds on the A100 (modes_test runs cpu, gpu and hybrid in the second).
# 3. Speed of the second against the first, interleaved: rebuild, exact start and steps at 0.03 and 0.3 spacings.
# 4. CI's host steps: OpenMP build (gcc 15) with sanitizers off and on, ctest on 2 threads, clang-tidy 18.
# 5. backends_probe: the CPU's time on the clipping's tail cells.
#SBATCH -J modes
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=04:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
repo="$(pwd)"
out="$repo/build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 5 >"$out/clock.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
cuda() {
    module purge
    module load gcc/14 cuda/13.0 cmake
}
configure() {  # configure <dir> <kokkos>
    cmake -S "$repo" -B "$1" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$2" \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >"$1.configure.log" 2>&1
    echo "configure $(basename "$1") exit $?"
}
timed() {  # timed <dir> <target>: -j1, wall seconds and bytes
    local t0=$(date +%s)
    cmake --build "$1" --target "$2" -j1 >"$1.$2.log" 2>&1
    echo "build $(basename "$1") $2 exit $? seconds $(($(date +%s) - t0)) bytes $(stat -c %s "$1/$2" 2>/dev/null)"
    grep -m3 error "$1.$2.log"
}

echo "== 1. build cost"
cuda
configure "$out/main" /u/uli/kokkos-cuda
configure "$out/both" /u/uli/kokkos-cuda-omp
timed "$out/main" voronoi_bench &
p1=$!
timed "$out/both" voronoi_bench &
wait $p1 $!  # not a bare wait: the clock logger runs in the background too
timed "$out/main" mesh_test &
p1=$!
timed "$out/both" mesh_test &
wait $p1 $!
cmake --build "$out/main" -j8 >"$out/main.all.log" 2>&1 &
p1=$!
cmake --build "$out/both" -j8 >"$out/both.all.log" 2>&1 &
wait $p1 $!
echo "build all: main exit $(grep -c 'error' "$out/main.all.log") errors, both $(grep -c 'error' "$out/both.all.log") errors"
grep -m5 error "$out/both.all.log"

echo "== 2. A100 ctest"
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ctest --test-dir "$out/main" --output-on-failure >"$out/ctest_main.log" 2>&1
echo "ctest main: $(grep -E 'tests passed|tests failed' "$out/ctest_main.log")"
ctest --test-dir "$out/both" --output-on-failure >"$out/ctest_both.log" 2>&1
echo "ctest both: $(grep -E 'tests passed|tests failed' "$out/ctest_both.log")"
"$out/both/modes_test" | grep -E "space|ok|FAIL"

echo "== 3. speed, both against main, interleaved"
for round in 1 2 3; do
    for kind in jittered uniform clustered halo cosmicweb; do
        for b in main both; do
            "$out/$b/voronoi_bench" rebuild 2097152 "$kind" 5 --no-check | grep "^warp:" | sed "s/^/round=$round build=$b kind=$kind n=2097152 /"
        done
        for d in 0.03 0.3; do
            for b in main both; do
                "$out/$b/voronoi_bench" step 2097152 "$kind" "$d" 10 | grep -E "^(mesh|gpuflips):" |
                    sed "s/^/round=$round build=$b kind=$kind n=2097152 d=$d /"
            done
        done
    done
    for b in main both; do
        "$out/$b/voronoi_bench" rebuild 16777216 jittered 3 --no-check | grep "^warp:" | sed "s/^/round=$round build=$b kind=jittered n=16777216 /"
        "$out/$b/voronoi_bench" step 16777216 jittered 0.03 6 | grep -E "^(mesh|gpuflips):" |
            sed "s/^/round=$round build=$b kind=jittered n=16777216 d=0.03 /"
    done
done >"$out/speed.log"
echo "speed lines: $(wc -l <"$out/speed.log")"

echo "== 4. CI host steps"
module purge
module load gcc/15 cmake
builds=()
for san in OFF ON; do
    cmake -S "$repo" -B "$out/host-$san" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-gcc15 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVORONOI_SANITIZERS=$san >/dev/null 2>&1
    cmake --build "$out/host-$san" -j9 >"$out/host-$san.build.log" 2>&1 &
    builds+=($!)
done
wait "${builds[@]}"
for san in OFF ON; do
    OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1 \
        ctest --test-dir "$out/host-$san" --output-on-failure --parallel 2 >"$out/ctest_host_$san.log" 2>&1
    echo "ctest host sanitizers=$san: $(grep -E 'tests passed|tests failed' "$out/ctest_host_$san.log")"
done
module load clang/18
clang-tidy -p "$out/host-OFF" "$repo/src/test/modes_test.cpp" >"$out/tidy_modes.log" 2>&1
echo "clang-tidy modes_test (includes voronoi_engine.hpp) exit $? warnings $(grep -c 'warning:' "$out/tidy_modes.log")"
grep -E "voronoi_engine|modes_test" "$out/tidy_modes.log" | grep -E "warning|error" | head -20

echo "== 5. backends_probe"
cuda
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for row in "uniform 33" "clustered 68" "halo 261" "cosmicweb 976"; do
    read -r kind tail <<<"$row"
    echo "-- $kind 128"
    timeout 1500 "$out/both/backends_probe" 2097152 "$kind" 0.1 2 "$tail" | grep -E "^(clip|tail|same|backend)"
done
