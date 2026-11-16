#!/bin/bash
# The rounding as a run-time choice (Layout::same_bits_as_host; hybrid turns it on), on one gpu1 node, from the
# repository root of branch cpu-gpu-modes:
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/modes_rounding_a100.sh <main checkout> <bits2d source dir>
# 1. Builds: main (CUDA) and this branch (CUDA; CUDA + OpenMP); voronoi_bench at -j1 side by side (compile time, size).
# 2. A100 ctest of this branch in both builds; modes_test (hybrid must equal cpu in every bit).
# 3. gpu mode against main, bit for bit: voronoi_bench --dump-cells (3D, five inputs, 128³) and bits2d (2D).
# 4. backends_probe: hybrid's arithmetic against the host, and the clipping's cost of it.
# 5. Speed, interleaved: main, this branch (gpu), this branch --same-bits (hybrid's arithmetic): rebuild, exact start,
#    steps at 0.03, 0.1, 0.3; 128³ (five inputs) and 256³ (jittered, cosmicweb).
# 6. CI host steps: OpenMP build (gcc 15), sanitizers off and on, ctest on 2 threads; clang-tidy 18 over the programs.
#SBATCH -J modes_rounding
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=07:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
here="$(pwd)"
main=$(realpath "$1") bits2d=$(realpath "$2")
out="$here/build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out/dump"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm --format=csv,noheader
cuda() {
    module purge
    module load gcc/14 cuda/13.0 cmake
}
host() {
    module purge
    module load gcc/15 cmake
}
configure() {  # configure <source> <dir> <kokkos>
    cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$3" \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >"$2.configure.log" 2>&1
}
timed() {  # timed <dir> <target>: -j1, wall seconds and bytes
    local t0=$(date +%s)
    cmake --build "$1" --target "$2" -j1 >"$1.$2.log" 2>&1
    echo "build $(basename "$1") $2 exit $? seconds $(($(date +%s) - t0)) bytes $(stat -c %s "$1/$2" 2>/dev/null)"
}
gcc_build() {  # gcc_build <dir> <sanitizers>
    cmake -S "$here" -B "$1" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-gcc15 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVORONOI_SANITIZERS="$2" >"$1.configure.log" 2>&1
    cmake --build "$1" -j16 >"$1.build.log" 2>&1
    echo "build $(basename "$1") exit $? errors $(grep -c ' error' "$1.build.log")"
}

echo "== 1. builds"
cuda
configure "$main" "$out/main" /u/uli/kokkos-cuda
configure "$here" "$out/cuda" /u/uli/kokkos-cuda
configure "$here" "$out/both" /u/uli/kokkos-cuda-omp
timed "$out/main" voronoi_bench &
p1=$!
timed "$out/cuda" voronoi_bench &
wait $p1 $!
for b in main cuda both; do
    cmake --build "$out/$b" -j16 >"$out/$b.build.log" 2>&1
    echo "build $b all exit $? errors $(grep -c ' error' "$out/$b.build.log")"
done
for b in main cuda; do
    src=$main
    [[ $b == cuda ]] && src=$here
    cmake -S "$bits2d" -B "$out/bits2d-$b" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF -DVORONOI_SRC="$src" \
        >"$out/bits2d-$b.configure.log" 2>&1
    cmake --build "$out/bits2d-$b" -j4 >"$out/bits2d-$b.build.log" 2>&1
    echo "build bits2d-$b exit $?"
done

echo "== 2. A100 ctest, modes_test"
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for b in cuda both; do
    ctest --test-dir "$out/$b" --output-on-failure >"$out/ctest_$b.log" 2>&1
    echo "ctest $b: $(grep -E 'tests passed|tests failed' "$out/ctest_$b.log")"
    grep -E "^ *[0-9]+ - " "$out/ctest_$b.log"
done
"$out/both/modes_test" >"$out/modes.log" 2>&1
echo "modes_test exit $?: $(grep -c '^ *ok' "$out/modes.log") ok, $(grep -c '^ *FAIL' "$out/modes.log") FAIL"
grep -E "FAIL|hybrid equals|gpu equals" "$out/modes.log" | head -24

echo "== 3. gpu mode against main"
for kind in uniform clustered halo cosmicweb jittered; do
    for b in main cuda; do
        "$out/$b/voronoi_bench" step 2097152 "$kind" 0.1 3 --dump-cells="$out/dump/$b-$kind" >/dev/null 2>&1
    done
    if cmp -s "$out/dump/main-$kind.geometry.npy" "$out/dump/cuda-$kind.geometry.npy" &&
        cmp -s "$out/dump/main-$kind.neighbours.npy" "$out/dump/cuda-$kind.neighbours.npy"; then
        echo "3D $kind: gpu mode equals main in every bit"
    else
        echo "3D $kind: gpu mode DIFFERS from main"
    fi
done
for b in main cuda; do "$out/bits2d-$b/bits2d" 1048576 4 >"$out/bits2d-$b.txt" 2>&1; done
if cmp -s "$out/bits2d-main.txt" "$out/bits2d-cuda.txt"; then
    echo "2D: gpu mode equals main in every bit ($(wc -l <"$out/bits2d-main.txt") updates)"
else
    echo "2D: gpu mode DIFFERS from main"
    paste "$out/bits2d-main.txt" "$out/bits2d-cuda.txt" | head
fi

echo "== 4. backends_probe"
for kind in uniform clustered halo cosmicweb; do
    echo "-- $kind 128"
    timeout 1800 "$out/both/backends_probe" 2097152 "$kind" 0.1 2 0 | grep -E "^(clip_kernels|clip_rounded|same|volume_error)"
done

echo "== 5. speed, interleaved"
run() {  # run <round> <name> <dir> <n> <kind> <steps> [flags]
    local round=$1 name=$2 dir=$3 n=$4 kind=$5 steps=$6
    shift 6
    "$out/$dir/voronoi_bench" rebuild "$n" "$kind" 5 --no-check "$@" | grep "^warp:" |
        sed "s/^/round=$round build=$name kind=$kind n=$n /"
    for d in 0.03 0.1 0.3; do
        "$out/$dir/voronoi_bench" step "$n" "$kind" "$d" "$steps" "$@" | grep -E "^(mesh|gpuflips):" |
            sed "s/^/round=$round build=$name kind=$kind n=$n d=$d /"
    done
}
for round in 1 2 3; do
    for row in "2097152 jittered 10" "2097152 uniform 10" "2097152 clustered 10" "2097152 halo 10" \
        "2097152 cosmicweb 10" "16777216 jittered 6" "16777216 cosmicweb 6"; do
        read -r n kind steps <<<"$row"
        run $round main main "$n" "$kind" "$steps"
        run $round gpu cuda "$n" "$kind" "$steps"
        run $round same_bits cuda "$n" "$kind" "$steps" --same-bits
    done
done >"$out/speed.log"
echo "speed lines: $(wc -l <"$out/speed.log")"

echo "== 6. CI host steps"
host
for san in OFF ON; do gcc_build "$out/host-$san" "$san"; done
for san in OFF ON; do
    OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1 \
        ctest --test-dir "$out/host-$san" --output-on-failure --parallel 2 >"$out/ctest_host_$san.log" 2>&1
    echo "ctest host sanitizers=$san: $(grep -E 'tests passed|tests failed' "$out/ctest_host_$san.log")"
done
module load clang/18
bash scripts/lint_cpp.sh "$out/host-OFF" >"$out/tidy.log" 2>&1
echo "clang-tidy 18: exit $? warnings $(grep -c 'warning:' "$out/tidy.log")"
grep -E "warning:|error:" "$out/tidy.log" | grep -v "^/u/uli/kokkos" | head -20
