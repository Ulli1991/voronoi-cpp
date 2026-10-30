#!/bin/bash
# Hydro stage 1 on one A100: the Sedov validation runs (ours and LUNA's) or
# the throughput and memory measurements. Submit from the repository root
# with a built CUDA directory and an output directory on a shared file system:
#   sbatch -o <logs>/hydro-%j.out scripts/hydro_a100.sh <build-cuda> <out> validate|throughput
# LUNA: LUNA_DIR (default /raven/u/uli/LUNA) with its environment
# LUNA_ENV (default ~/jax-env), as LUNA's submit/ scripts use them.
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
    "$build/hydro_sedov" "$@"
}

luna() {  # luna <order> <run dir> [Config overrides]
    local order="$1" dir="$2"
    shift 2
    module purge
    module load gcc/13 cuda/12
    (cd "$luna_dir" && "$luna_env/bin/python" -u -m scripts.validation.runs.sedov3d_run nx=128 ORDER="$order" LIMITER=0 \
        riemann_solver=hll enable_x64=1 output_dir="$dir" "$@" 2>&1 |
        grep -E "Sedov|inject|Performance|Total steps|Total runtime|ms/step|Error|error" | grep -v "^\s*$")
}

if [[ "$mode" == validate ]]; then
    module purge
    module load gcc/14 cuda/13.0 cmake hdf5-serial
    ctest --test-dir "$build" --output-on-failure
    for order in 1 0; do
        ours 128 jittered --order=$order --out="$out/voronoi128_o$order" | grep -v "^  "
    done
    ours 128 uniform --order=1 --out="$out/voronoi128_uniform_o1" | grep -v "^  "
    for order in 1 0; do
        echo "== LUNA ORDER=$order hll nx=128"
        luna $order "$out/luna128_o$order"
    done
elif [[ "$mode" == throughput ]]; then
    for n in 128 256; do
        for order in 1 0; do
            for update in cell face; do
                echo "== ours n=${n}^3 jittered order=$order update=$update"
                ours $n jittered --order=$order --update=$update --steps=40 --memory
            done
        done
        for update in cell face; do
            echo "== ours n=${n}^3 jittered order=1 update=$update input order"
            ours $n jittered --order=1 --update=$update --steps=40 --input-order
        done
    done
    for update in cell face; do
        echo "== ours n=128^3 uniform order=1 update=$update"
        ours 128 uniform --order=1 --update=$update --steps=40 --memory
        echo "== ours n=128^3 uniform order=1 update=$update input order"
        ours 128 uniform --order=1 --update=$update --steps=40 --input-order
    done
    for order in 1 0; do
        echo "== LUNA timing ORDER=$order hll nx=128 (no snapshots)"
        luna $order "$out/luna128_timing_o$order" write_snapshots=0
    done
fi
