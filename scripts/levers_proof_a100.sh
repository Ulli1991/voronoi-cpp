#!/bin/bash
# The proof of the step levers (branch step-levers) in one gpu1 job, against step-2x (ref): fresh builds of both, A100
# ctest of the branch (modes: hybrid equals cpu), every step against a fresh exact build on seven inputs at three
# spacings and S2's jump cases, geometry, faces and lists checksums against ref, then CI's host steps. From the branch:
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_proof_a100.sh <ref checkout> <base ref for the format check> <source>...
# The sources are the changed programs for clang-tidy.
#SBATCH -J lev_proof
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=03:00:00
set -uo pipefail
ref_src=$(realpath "$1") base="$2"
shift 2
here=$(pwd)
bin="$here/build-jobs/levers/proof-$SLURM_JOB_ID"
mkdir -p "$bin"
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics --format=csv,noheader
echo "== build: ref $(git -C "$ref_src" log --oneline -1 | cut -c1-60), new $(git log --oneline -1 | cut -c1-60)$(git diff --quiet HEAD -- src || echo ' +changes')"
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
run() {  # run <tag> <args...>: one bench run, its lines prefixed
    local tag="$1"
    shift
    timeout 600 "$bin/voronoi_bench_$tag" "$@" --no-start 2>&1 |
        grep -E "^(gpuflips|geometry_checksum|verify|refused)" | sed "s#^#$tag $* | #"
}
echo "== verify: every step against a fresh exact build"
for kind in jittered uniform clustered halo lattice wave cosmicweb; do
    for d in 0.03 0.1 0.3; do run new step 262144 "$kind" "$d" 4 --verify; done
done
for d in 0.03 0.1 0.3; do run new step 2097152 jittered "$d" 3 --verify; done
echo "== verify: S2's jump cases"
STEPS=6 bash scripts/local_repair_a100.sh "$bin" verify 2097152 2>&1 | grep -E "^(== case|verify|refused)"
echo "== bits"
for kind in jittered uniform clustered halo lattice wave cosmicweb; do
    for d in 0.03 0.1 0.3; do
        for tag in ref new; do run "$tag" step 2097152 "$kind" "$d" 5 --checksum; done
    done
done
nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader
echo "== ci host steps"
cd "$here" && bash scripts/ci_host_general.sh "$base" "$@"
