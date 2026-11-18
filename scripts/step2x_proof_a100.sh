#!/bin/bash
# The whole step-2x proof in one gpu1 job, against main (ref) as S2's proof did: fresh builds of both, A100 ctest of
# the branch, S2's verified jump cases and jump rates, step rates (128³, 256³), checksums (seven inputs), 500-step runs,
# cells by particle after 500 steps, then CI's host steps. From the branch's checkout:
#   sbatch -o build-jobs/logs/%x-%j.out scripts/step2x_proof_a100.sh <ref checkout> <base ref for the format check>
#SBATCH -J step2x_proof
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=10:00:00
set -uo pipefail
ref_src=$(realpath "$1") base="$2"
here=$(pwd)
bin="$here/build-jobs/step2x/proof-$SLURM_JOB_ID"
mkdir -p "$bin/dump"
cuda() {
    source /etc/profile.d/modules.sh
    module purge
    module load gcc/14 cuda/13.0 cmake
}
cuda
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 5 >"$bin/clock.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
echo "== build: ref $(git -C "$ref_src" log --oneline -1 | cut -c1-60), new $(git log --oneline -1 | cut -c1-60)"
for side in ref new; do
    src=$here
    [[ $side == ref ]] && src=$ref_src
    cmake -S "$src" -B "$bin/build-$side" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >"$bin/$side.configure.log" 2>&1
done
cmake --build "$bin/build-ref" --target voronoi_bench --parallel 6 >"$bin/ref.build.log" 2>&1 &
cmake --build "$bin/build-new" --parallel 12 >"$bin/new.build.log" 2>&1
wait $!
grep -E " error|error:" "$bin/ref.build.log" "$bin/new.build.log" | head
cp "$bin/build-ref/voronoi_bench" "$bin/voronoi_bench_ref"
cp "$bin/build-new/voronoi_bench" "$bin/voronoi_bench_new"
echo "== ctest (new)"
ctest --test-dir "$bin/build-new" --output-on-failure 2>&1 | tail -40
part() {  # S2's parts, ref and new from $bin
    echo "== part $1 $2"
    STEPS=${3:-} bash scripts/local_repair_a100.sh "$bin" "$1" "$2"
}
part verify 2097152 6
part speed 2097152 12
run() {  # run <tag> <args...>: one bench run, its lines prefixed
    local tag="$1"
    shift
    "$bin/voronoi_bench_$tag" "$@" --no-start 2>&1 |
        grep -E "^(gpuflips|geometry_checksum|mesh|final_rebuild|layout|verify)" | sed "s#^#$tag $* | #"
}
echo "== rates"
for n in 2097152 16777216; do
    for kind in jittered uniform clustered halo; do
        for d in 0.03 0.1 0.3; do
            for tag in ref new ref new; do run "$tag" step "$n" "$kind" "$d" 10; done
        done
    done
done
echo "== bits"
for kind in jittered uniform clustered halo lattice wave cosmicweb; do
    for d in 0.03 0.1 0.3; do
        for tag in ref new; do run "$tag" step 2097152 "$kind" "$d" 5 --checksum; done
    done
done
echo "== long"
for spec in "2097152 0.1" "16777216 0.1" "2097152 0.03" "2097152 0.3"; do
    set -- $spec
    for tag in ref new; do run "$tag" step "$1" jittered "$2" 500; done
done
echo "== dump"
for tag in ref new; do run "$tag" step 2097152 jittered 0.1 500 --dump-cells="$bin/dump/$tag"; done
ls -l "$bin/dump"
nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader
echo "== ci host steps"
kill $smi 2>/dev/null
cd "$here" && bash scripts/ci_host_general.sh "$base" src/bench/voronoi_bench.cpp src/test/order_test.cpp \
    src/test/modes_test.cpp src/test/mesh_test.cpp src/test/mesh_cases/fallback.cpp src/test/repair_test.cpp
