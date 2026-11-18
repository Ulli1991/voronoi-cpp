#!/bin/bash
# The mesh's geometry without fused multiply-add on the device (option c of TESTING.md, "Bits across backends"), on
# one gpu1 node, from the cpu-overlap repository root:
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/unfused_a100.sh <main> <unfused>
# <main>: a checkout of main (8cb1475); <unfused>: the same with the geometry unfused (branch unfused-on-main).
# 1. Builds: CUDA (Kokkos CUDA) and OpenMP (gcc 15) of both checkouts; this branch on Kokkos CUDA + OpenMP.
# 2. A100 ctest of <unfused> and of this branch (modes_test: cpu, gpu and hybrid must give the same bits).
# 3. Bits: per-cell volumes and centroids after 3 steps (voronoi_bench --dump-cells) of <unfused> on CUDA against
#    OpenMP, and of <unfused> against <main> on CUDA (the default path's change) and on OpenMP; backends_probe here.
# 4. Speed, interleaved: <unfused> against <main>, rebuild, exact start, steps at 0.03, 0.1, 0.3; 128³ and 256³.
# 5. CI host steps on <unfused> and here: OpenMP build (gcc 15), sanitizers off and on, ctest on 2 threads;
#    clang-tidy 18 over <unfused>'s programs (scripts/lint_cpp.sh) and on this branch's modes_test.
#SBATCH -J unfused
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=06:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
here="$(pwd)"
main=$(realpath "$1") unfused=$(realpath "$2")
out="$here/build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out"
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
nvcc_build() {  # nvcc_build <source> <dir> <kokkos>
    cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$3" \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >"$2.configure.log" 2>&1
    cmake --build "$2" -j16 >"$2.build.log" 2>&1
    echo "build $2 exit $? errors $(grep -c ' error' "$2.build.log")"
}
gcc_build() {  # gcc_build <source> <dir> <sanitizers>
    cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-gcc15 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVORONOI_SANITIZERS="$3" >"$2.configure.log" 2>&1
    cmake --build "$2" -j16 >"$2.build.log" 2>&1
    echo "build $2 exit $? errors $(grep -c ' error' "$2.build.log")"
}

echo "== 1. builds"
cuda
nvcc_build "$main" "$out/main-cuda" /u/uli/kokkos-cuda
nvcc_build "$unfused" "$out/unfused-cuda" /u/uli/kokkos-cuda
nvcc_build "$here" "$out/branch-both" /u/uli/kokkos-cuda-omp
host
gcc_build "$main" "$out/main-omp" OFF
for san in OFF ON; do
    gcc_build "$unfused" "$out/unfused-host-$san" "$san"
    gcc_build "$here" "$out/branch-host-$san" "$san"
done

echo "== 2. A100 ctest"
cuda
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for b in unfused-cuda branch-both; do
    ctest --test-dir "$out/$b" --output-on-failure >"$out/ctest_$b.log" 2>&1
    echo "ctest $b: $(grep -E 'tests passed|tests failed' "$out/ctest_$b.log")"
    grep -E "^ *[0-9]+ - " "$out/ctest_$b.log"
done
"$out/branch-both/modes_test" >"$out/modes.log" 2>&1
echo "modes_test exit $?: $(grep -c '^ *ok' "$out/modes.log") ok, $(grep -c '^ *FAIL' "$out/modes.log") FAIL"
grep -E "FAIL|equals cpu" "$out/modes.log" | grep -E "FAIL|gpu equals" | head -20

echo "== 3. bits"
mkdir -p "$out/dump"
for kind in uniform clustered halo cosmicweb jittered; do
    for b in main-cuda unfused-cuda; do
        "$out/$b/voronoi_bench" step 2097152 "$kind" 0.1 3 --dump-cells="$out/dump/$b-$kind" >/dev/null 2>&1
    done
done
host
for kind in uniform clustered halo cosmicweb jittered; do
    for b in main-omp unfused-host-OFF; do
        "$out/$b/voronoi_bench" step 2097152 "$kind" 0.1 3 --dump-cells="$out/dump/$b-$kind" >/dev/null 2>&1
    done
done
ls "$out/dump" | head -40
cuda
for kind in uniform clustered halo cosmicweb; do
    echo "-- probe $kind 128"
    timeout 1500 "$out/branch-both/backends_probe" 2097152 "$kind" 0.1 2 0 | grep -E "^(clip_kernels|same|volume_)"
done

echo "== 4. speed, interleaved"
for round in 1 2 3; do
    for row in "2097152 jittered" "2097152 uniform" "2097152 clustered" "2097152 halo" "2097152 cosmicweb" \
        "16777216 jittered" "16777216 cosmicweb"; do
        read -r n kind <<<"$row"
        steps=10
        [[ $n == 16777216 ]] && steps=6
        for b in main-cuda unfused-cuda; do
            "$out/$b/voronoi_bench" rebuild "$n" "$kind" 5 --no-check | grep "^warp:" |
                sed "s/^/round=$round build=$b kind=$kind n=$n /"
        done
        for d in 0.03 0.1 0.3; do
            for b in main-cuda unfused-cuda; do
                "$out/$b/voronoi_bench" step "$n" "$kind" "$d" "$steps" | grep -E "^(mesh|gpuflips):" |
                    sed "s/^/round=$round build=$b kind=$kind n=$n d=$d /"
            done
        done
    done
done >"$out/speed.log"
echo "speed lines: $(wc -l <"$out/speed.log")"

echo "== 5. CI host steps"
host
for b in unfused branch; do
    for san in OFF ON; do
        OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1 \
            ctest --test-dir "$out/$b-host-$san" --output-on-failure --parallel 2 >"$out/ctest_$b-host-$san.log" 2>&1
        echo "ctest $b host sanitizers=$san: $(grep -E 'tests passed|tests failed' "$out/ctest_$b-host-$san.log")"
    done
done
module load clang/18
(cd "$unfused" && bash scripts/lint_cpp.sh "$out/unfused-host-OFF") >"$out/tidy_unfused.log" 2>&1
echo "clang-tidy 18 unfused: exit $? warnings $(grep -c 'warning:' "$out/tidy_unfused.log")"
grep -E "warning:|error:" "$out/tidy_unfused.log" | grep -v "^/u/uli/kokkos" | head -20
clang-tidy -p "$out/branch-host-OFF" "$here/src/test/modes_test.cpp" >"$out/tidy_branch.log" 2>&1
echo "clang-tidy 18 branch modes_test: exit $? warnings $(grep -c 'warning:' "$out/tidy_branch.log")"
