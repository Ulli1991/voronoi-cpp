#!/bin/bash
# The 2x step work (branch step-2x) on one A100: builds voronoi_bench of each named tree, then measures them in turn.
#   sbatch -o build-jobs/logs/%x-%j.out scripts/step2x_a100.sh <mode> <name>=<source> [<name>=<source> ...] [-- args]
# Modes: kernels (per-kernel ms, jittered and uniform 128³), rates (128³ and 256³, four inputs, three spacings,
# rounds interleaved), long (500-step smooth flow; args n, spacings, flag sets separated by ';'), bits
# (checksums, seven inputs, 0.03 / 0.1 / 0.3, 128³), dump (per-particle cells after 500 steps; flag sets as long).
#SBATCH -J step2x
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
here="$(pwd)"
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
out="$here/build-jobs/step2x"
bin="$out/bin-$SLURM_JOB_ID"
mkdir -p "$bin"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 5 >"$bin/clock.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
if [[ "$mode" == ctest ]]; then  # every target of the first tree, ctest (fuzz smoke included), then the fuzz batch
    dir="$bin/build-${names[0]}"
    cmake -S "${src[1]}" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >"$dir.configure.log" 2>&1
    cmake --build "$dir" --parallel 18 2>&1 | grep -E " error|Error " | head -20
    echo "build ${names[0]} ($(git -C "${src[1]}" rev-parse --short HEAD))"
    ctest --test-dir "$dir" --output-on-failure 2>&1 | tail -60
    [[ ${#args[@]} -gt 0 ]] && "$dir/mesh_fuzz_test" "${args[@]}" 2>&1 | tail -30
    exit 0
fi
k=0
builds=()
for name in "${names[@]}"; do
    k=$((k + 1))
    dir="$bin/build-$name"
    (
        cmake -S "${src[$k]}" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
            -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF -DBUILD_TESTING=OFF \
            >"$dir.configure.log" 2>&1
        cmake --build "$dir" --target voronoi_bench -j4 >"$dir.build.log" 2>&1
        rc=$?
        echo "build $name ($(git -C "${src[$k]}" rev-parse --short HEAD)$(git -C "${src[$k]}" diff --quiet || echo +)) exit $rc"
        cp "$dir/voronoi_bench" "$bin/voronoi_bench_$name"
    ) &
    builds+=($!)
done
wait "${builds[@]}"  # not the clock logger
run() {  # run <name> <args...>: one bench run, its lines prefixed by the name and arguments
    local name="$1"
    shift
    "$bin/voronoi_bench_$name" "$@" 2>&1 | grep -E "^(gpuflips|geometry_checksum|mesh|kernel|final_rebuild|layout)" |
        sed "s/^/$name $* | /"
}
case "$mode" in
kernels)
    for kind in jittered uniform; do
        for d in 0.03 0.1 0.3; do
            for name in "${names[@]}"; do run "$name" step 2097152 "$kind" "$d" 10 --kernels "${args[@]}"; done
        done
    done
    ;;
rates)
    for n in 2097152 16777216; do
        for kind in jittered uniform clustered halo; do
            for d in 0.03 0.1 0.3; do
                for rep in 1 2; do
                    for name in "${names[@]}"; do run "$name" step "$n" "$kind" "$d" 10 "${args[@]}"; done
                done
            done
        done
    done
    ;;
long)  # args: <n> <spacings> <flag sets>; every name runs the first set, the last name runs every set
    n="${args[0]}" d="${args[1]}"
    IFS=';' read -ra sets <<<"${args[2]}"
    last="${names[-1]}"
    for name in "${names[@]}"; do
        # shellcheck disable=SC2086
        run "$name" step "$n" jittered "$d" 500 ${sets[0]}
    done
    for flags in "${sets[@]:1}"; do
        # shellcheck disable=SC2086
        run "$last" step "$n" jittered "$d" 500 $flags
    done
    ;;
bits)
    for kind in jittered uniform clustered halo lattice wave cosmicweb; do
        for d in 0.03 0.1 0.3; do
            for name in "${names[@]}"; do run "$name" step 2097152 "$kind" "$d" 5 --checksum "${args[@]}"; done
        done
    done
    ;;
dump)
    n="${args[0]}"
    IFS=';' read -ra sets <<<"${args[1]}"
    k=0
    for flags in "${sets[@]}"; do
        k=$((k + 1))
        for name in "${names[@]}"; do
            # shellcheck disable=SC2086
            run "$name" step "$n" jittered 0.1 500 $flags --dump-cells="$bin/$name-$k"
        done
    done
    ls -l "$bin"/*.npy
    ;;
*)
    echo "unknown mode $mode"
    exit 2
    ;;
esac
nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader
