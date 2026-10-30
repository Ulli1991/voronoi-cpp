#!/bin/bash
# Hydro stage 1 on one A100. Submit from the repository root with a built
# CUDA directory and an output directory on a shared file system:
#   sbatch -o <logs>/hydro-%j.out scripts/hydro_a100.sh <build-cuda> <out> <mode>
# Modes:
#   validate     ctest; Sedov 128^3 with snapshots, ours and LUNA, HLLC and
#                HLL, second and first order
#   limiters     ours, Sedov 128^3 with snapshots, HLLC, the limiter at face
#                centroids, at cell centroids, none; uniform random points
#   throughput   ours at 128^3 (both loops, both orders, both solvers,
#                --memory; whole MUSCL-Hancock runs) and LUNA at 128^3
#                (XLA and Pallas kernels, whole runs), no snapshots
#   throughput256  the same at 256^3 (LUNA: 100 steps)
# LUNA: LUNA_DIR (default /raven/u/uli/LUNA) with its environment LUNA_ENV
# (default ~/jax-env), as LUNA's submit/ scripts use them; FP64, minmod.
#SBATCH -J hydro_a100
#SBATCH -p gpu1,gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
build="$(realpath "${1:-build-cuda}")"
out="$(realpath -m "${2:-hydro-runs}")"
mode="${3:-validate}"
luna_dir="${LUNA_DIR:-/raven/u/uli/LUNA}"
luna_env="${LUNA_ENV:-$HOME/jax-env}"
mkdir -p "$out"
nvidia-smi -L

ours() {
    module purge
    module load gcc/14 cuda/13.0 cmake hdf5-serial
    "$build/hydro_sedov" "$@" | grep -v "^  "
}

luna() {  # luna <nx> <order> <riemann solver> <run dir> [Config overrides]
    local nx="$1" order="$2" solver="$3" dir="$4"
    shift 4
    module purge
    module load gcc/13 cuda/12
    (cd "$luna_dir" && "$luna_env/bin/python" -u -m scripts.validation.runs.sedov3d_run nx="$nx" ORDER="$order" LIMITER=0 \
        riemann_solver="$solver" enable_x64=1 output_dir="$dir" "$@" 2>&1 |
        grep -E "Sedov 3D|inject|perf =|Performance|Total steps|Total runtime|Error|error")
}

if [[ "$mode" == validate ]]; then
    module purge
    module load gcc/14 cuda/13.0 cmake hdf5-serial
    ctest --test-dir "$build" --output-on-failure
    for solver in hllc hll; do
        for order in 1 0; do
            echo "== ours 128^3 jittered order=$order $solver"
            ours 128 jittered --order=$order --riemann=$solver --out="$out/voronoi128_o${order}_$solver"
            echo "== LUNA 128^3 ORDER=$order $solver"
            luna 128 $order $solver "$out/luna128_o${order}_$solver"
        done
    done
elif [[ "$mode" == limiters ]]; then
    for limiter in face cell none; do
        echo "== ours 128^3 jittered order=1 hllc limiter=$limiter"
        ours 128 jittered --order=1 --limiter=$limiter --out="$out/voronoi128_o1_hllc_$limiter"
    done
    echo "== ours 128^3 uniform order=1 hllc"
    ours 128 uniform --order=1 --out="$out/voronoi128_uniform_o1_hllc"
elif [[ "$mode" == throughput || "$mode" == throughput256 ]]; then
    n=128
    luna_steps=()
    if [[ "$mode" == throughput256 ]]; then
        n=256
        luna_steps=(max_steps=100)
    fi
    for solver in hllc hll; do
        for order in 1 0; do
            for update in face cell; do
                echo "== ours n=${n}^3 jittered order=$order update=$update $solver"
                ours $n jittered --order=$order --update=$update --riemann=$solver --steps=40 --memory
            done
        done
    done
    if [[ "$mode" == throughput ]]; then
        for solver in hllc hll; do
            echo "== ours 128^3 jittered order=1 $solver, whole run, no snapshots"
            ours 128 jittered --order=1 --riemann=$solver
        done
    fi
    for solver in hllc hll; do
        for kernel in hydro_kernel=pallas hydro_kernel=xla; do
            echo "== LUNA ${n}^3 ORDER=1 $solver $kernel, no snapshots"
            luna $n 1 $solver "$out/luna${n}_timing" write_snapshots=0 $kernel "${luna_steps[@]}"
        done
    done
fi
