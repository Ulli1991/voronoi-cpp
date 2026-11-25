#!/bin/bash
# Every A100 number the README quotes, re-measured on the tree it runs from in one gpu1 job (issue #6): the test suite;
# full rebuilds and exact starts of six inputs at 128³ and 256³; the moving step, ten steps (steps 2-10, two rounds) and
# 500 steps of the bench flow; peak memory of a rebuild and of a step with and without a re-sort; the 2D mesh at 2²⁰ and
# 2²⁴; voronoi-cpp's rows of the library benchmark (every path, the shared point files, the moving trajectory). Tables:
# scripts/step2x_tables.py for the steps; the lines are kept in <out>.
#   sbatch -o <log dir>/%x-%j.out scripts/readme_numbers_a100.sh <build dir> <library data dir> <out dir>
#SBATCH -J readme_nums
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
build="$1"
data="$2"
out="$3"
mkdir -p "$out" "$data/volumes"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics,clocks.mem --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 5 >"$out/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
echo "== source $(git log --oneline -1 | cut -c1-70)$(git diff --quiet HEAD -- src CMakeLists.txt || echo ' +changes')"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >"$out/configure.log" 2>&1
cmake --build "$build" --parallel 18 >"$out/build.log" 2>&1
echo "build exit $?"
grep -E " error|error:" "$out/build.log" | head
bench="$build/voronoi_bench"
lines="$out/lines-$SLURM_JOB_ID.txt"
run() {  # run <args...>: one voronoi_bench run, its result lines prefixed by the arguments
    timeout 900 "$bench" "$@" 2>&1 | grep -E "^(gpuflips|start|rebuild|warp|mesh|memory|refused|kernel|layout)" |
        sed "s#^#$* | #" | tee -a "$lines"
}
echo "== ctest"
ctest --test-dir "$build" --output-on-failure 2>&1 | tail -30
echo "== rebuild"
for n in 2097152 16777216; do
    for kind in jittered uniform clustered halo lattice wave; do run rebuild "$n" "$kind" 5 --no-check; done
done
echo "== rebuild memory"
for n in 2097152 16777216; do run rebuild "$n" jittered 2 --no-check --memory; done
echo "== exact start"
for n in 2097152 16777216; do
    for kind in jittered uniform clustered halo lattice wave; do run step "$n" "$kind" 0.03 1; done
done
echo "== ten steps"
for n in 2097152 16777216; do
    for d in 0.03 0.1 0.3; do
        for rep in 1 2; do run step "$n" jittered "$d" 10 --no-start; done
    done
done
echo "== 500 steps"
for d in 0.03 0.1 0.3; do run step 2097152 jittered "$d" 500 --no-start; done
run step 16777216 jittered 0.1 500 --no-start
echo "== step memory"
for n in 2097152 16777216; do
    run step "$n" jittered 0.1 6 --no-start --memory
    run step "$n" jittered 0.1 6 --no-start --memory --every=3
done
echo "== 2D"
for n in 1048576 16777216; do
    for kind in jittered random clustered lattice; do
        timeout 600 "$build/mesh2d_bench" "$kind" "$n" --move=0.03 2>&1 | sed "s#^#2d $kind $n | #" | tee -a "$lines"
    done
done
echo "== library benchmark rows"
export LIBBENCH_COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src bench || echo +local)"
lib="$out/voronoi-cpp-a100-$SLURM_JOB_ID.jsonl"
for n in 2097152 16777216; do
    for kind in jittered uniform clustered halo bluenoise; do
        for path in warp warp-host warp-host-vc warp-vo warp-host-vo exact; do
            "$build/library_bench" build "$data/points/${kind}_${n}.pts" --path=$path --repeats=3 \
                --volumes="$data/volumes/voronoi-cpp-${path}-a100_${kind}_${n}.f64" >>"$lib" 2>>"$out/library.err"
        done
    done
done
for kind in jittered cosmicweb surface; do
    "$build/library_bench" step "$data/points/${kind}_2097152_step0.03.trj" \
        --volumes="$data/volumes/voronoi-cpp-step-a100_${kind}_2097152_step10.f64" >>"$lib" 2>>"$out/library.err"
done
echo "library records: $(grep -c . "$lib")"
nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader
