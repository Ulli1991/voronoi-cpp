#!/bin/bash
# Stage 1 (1D) on one A100, both codes in one job on one node. From the repository root:
#   sbatch --export=ALL,REPO=$PWD,BUILD=<build dir>,OUT=<runs dir> -o <logs>/dim1d-%j.out scripts/dim1d_a100.sh <mode> [args]
# Modes:
#   build            configure if needed, build everything wide (18 CPUs), ctest
#   ladder <problem> <b> [<b> ...]   both codes on N = 2^b: LUNA (MC, HLLC, CFL 0.8, Pallas), ours
#                    static and moving (LUNA's rule 0.8; AREPO's rule 0.3 up to 2^16), .npy/.npz + .json
#                    into OUT; repeats give the fastest; hashes must agree
#   gaps <problem> <b ...> -- <b ...>   moving with AREPO's rule, then LUNA's rule at 0.4, repeats 1
#   speed            throughput, BASELINE's rates, the smallest ladders
#   reference <problem> <N> <seconds>   LUNA MC at N for that long, resumable (state in OUT)
#   throughput       hydro only and whole moving step at 2^12 .. 2^24, ours and LUNA
# The clock and the node go into the log (nvidia-smi, one line per 5 s, into OUT/clock-<job>.csv).
#SBATCH -J dim1d
#SBATCH -p gpu1,gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
repo="${REPO:-$PWD}"
build="${BUILD:-/raven/u/uli/voronoi-cpp/build-1d}"
out="${OUT:-/raven/u/uli/voronoi-cpp/build-1d/runs}"
bin="${BIN:-$build/cuda}"  # a copy of the build, to keep it stable under a queue of jobs
mode="${1:-build}"
luna_dir="${LUNA_DIR:-/raven/u/uli/LUNA}"
luna_env="${LUNA_ENV:-$HOME/jax-env}"
mkdir -p "$out"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics,power.limit --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 >"$out/clock-${SLURM_JOB_ID:-0}.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT

use_ours() {
    module purge
    module load gcc/14 cuda/13.0 cmake hdf5-serial
}
use_luna() {
    module purge
    module load gcc/13 cuda/12
    export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false LUNA_REFERENCE_SOLUTIONS=/raven/u/uli/luna-reference-solutions
}
ours() {  # ours <problem> <N> <series> [flags]
    local p="$1" n="$2" s="$3"
    shift 3
    "$bin/hydro_1d" "$p" "$n" --memory --series="$s" --out="$out/${s}_${p}_${n}.npy" "$@" | grep -E "^(ours|memory|phase)"
}
luna() {  # luna <problem> <N> [key=value ...]
    local p="$1" n="$2"
    shift 2
    (cd "$out" && "$luna_env/bin/python" -u "$repo/scripts/luna_1d_run.py" "$p" "$n" hllc "$out/luna_${p}_${n}.npz" "$@" 2>&1 | grep -E "^luna|Error|error")
}

do_build() {
    use_ours
    [[ -f "$build/cuda/CMakeCache.txt" ]] || cmake -S "$repo" -B "$build/cuda" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper | tail -2
    cmake --build "$build/cuda" -j18 "$@" 2>&1 | grep -E "warning|error|Error |Built target" | sort | uniq -c | sort -rn | head -30
}
do_ladder() {
    local p="$1"
    shift
    for b in "$@"; do
        n=$((1 << b))
        r=$((b >= 15 ? 1 : 2))  # long runs once
        use_luna
        luna "$p" "$n" repeats=$r
        use_ours
        ours "$p" "$n" static --repeats=$r
        ours "$p" "$n" moving --moving --repeats=$r
        ((b <= 15)) && ours "$p" "$n" moving-arepo --moving --dt=arepo --repeats=1
    done
}
do_throughput() {
    use_ours
    for n in 4096 65536 1048576 16777216; do
        s=$((n > 2000000 ? 200 : 2000))
        "$bin/hydro_1d" sod "$n" --steps=$s --repeats=3 | grep -E "^ours"
        "$bin/hydro_1d" sod "$n" --steps=$s --repeats=3 --moving | grep -E "^ours"
        "$bin/hydro_1d" sod "$n" --steps=$s --repeats=2 --moving --phases | grep -E "^(ours|phase)"
    done
    use_luna
    for n in 4096 65536 1048576 16777216; do
        (cd "$out" && "$luna_env/bin/python" -u "$repo/scripts/luna_1d_run.py" sod "$n" hllc "$out/luna_rate_$n.npz" \
            steps=$((n > 2000000 ? 200 : 2000)) repeats=3 2>&1 | grep -E "^luna|^run|Error|error")
    done
}

case "$mode" in
build)
    do_build
    ctest --test-dir "$build/cuda" --output-on-failure | tail -14
    ;;
speed)  # throughput, an older binary's rates, the grid's bits on a moving Sod, the smallest ladders
    do_throughput
    use_ours
    if [[ -x "${BASELINE:-}" ]]; then
        for n in 4096 65536 1048576; do
            "$BASELINE" sod "$n" --steps=2000 --repeats=3 | grep -E "^ours" | sed 's/^/baseline /'
            "$BASELINE" sod "$n" --steps=2000 --repeats=3 --moving | grep -E "^ours" | sed 's/^/baseline /'
        done
    fi
    for p in sod shu-osher woodward-colella; do do_ladder "$p" 8 9 10 11 12; done
    ;;
ladder)
    do_ladder "$2" "${@:3}"
    ;;
gaps)  # moving at lower Courant numbers, above the ladders' sizes: gaps <problem> <arepo b ...> -- <0.4 b ...>
    use_ours
    p="$2"
    shift 2
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        ours "$p" "$((1 << $1))" moving-arepo --moving --dt=arepo --repeats=1
        shift
    done
    shift
    for b in "$@"; do ours "$p" "$((1 << b))" moving-c04 --moving --courant=0.4 --repeats=1; done
    ;;
reference)
    use_luna
    mkdir -p "$out/fine"
    (cd "$out/fine" && "$luna_env/bin/python" -u "$repo/scripts/luna_1d_run.py" "$2" "$3" hllc "$out/fine/$2_$3.npz" \
        series=reference max_s="$4" resume="$out/fine/state_$2_$3.npz" 2>&1 | grep -E "^luna|Error|error")
    ;;
throughput)
    do_throughput
    ;;
esac
