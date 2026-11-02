#!/bin/bash
# Stage 1 (1D) on one A100: LUNA's 1D tests, ours and LUNA's, at LUNA's
# resolution ladders, and throughput at large N. Submit from the
# repository root (logs on a shared file system):
#   sbatch -o <logs>/dim1d-%j.out scripts/dim1d_a100.sh <build-cuda> <out> <ours|luna>
# ours: ctest; hydro_1d for Sod (128-512), Shu-Osher (256-2048) and
#   Woodward-Colella (256-4096): static (HLLC, HLL; LUNA's dt rule at 0.8),
#   moving (HLLC, HLL at LUNA's rule 0.8; HLLC at AREPO's rule 0.3), and
#   static jittered (HLLC); then Sod for 200 steps at 2^20, 2^22, 2^24
#   cells, static and moving (throughput).
# luna: scripts/luna_1d_run.py for the same problems and sizes, MC
#   (LIMITER=1), HLLC and HLL, CFL 0.8, on the GPU (LUNA_DIR, default
#   /raven/u/uli/LUNA, environment LUNA_ENV, default ~/jax-env, as
#   hydro_a100.sh); then 200 steps of Sod at 2^20, 2^22, 2^24.
#SBATCH -J dim1d
#SBATCH -p gpu1,gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
build="$(realpath "${1:-build-cuda}")"
out="$(realpath -m "${2:-dim1d-runs}")"
mode="${3:-ours}"
repo="$(pwd)"
luna_dir="${LUNA_DIR:-/raven/u/uli/LUNA}"
luna_env="${LUNA_ENV:-$HOME/jax-env}"
mkdir -p "$out"
nvidia-smi -L
declare -A sizes=([sod]="128 256 512" [shu-osher]="256 512 1024 2048" [woodward-colella]="256 512 1024 2048 4096")
large="1048576 4194304 16777216"

if [[ "$mode" == ours ]]; then
    module purge
    module load gcc/14 cuda/13.0 cmake hdf5-serial
    ctest --test-dir "$build" --output-on-failure | tail -4
    for p in sod shu-osher woodward-colella; do
        for n in ${sizes[$p]}; do
            for r in hllc hll; do
                "$build/hydro_1d" "$p" "$n" --riemann=$r --out="$out/ours_${p}_${n}_static_$r.txt"
                "$build/hydro_1d" "$p" "$n" --riemann=$r --moving --out="$out/ours_${p}_${n}_moving_$r.txt"
            done
            "$build/hydro_1d" "$p" "$n" --moving --dt=arepo --out="$out/ours_${p}_${n}_moving-arepo_hllc.txt"
            "$build/hydro_1d" "$p" "$n" --jitter=0.5 --out="$out/ours_${p}_${n}_jittered_hllc.txt"
        done
    done
    for n in $large; do
        "$build/hydro_1d" sod "$n" --steps=200
        "$build/hydro_1d" sod "$n" --steps=200 --moving
        "$build/hydro_1d" sod "$n" --steps=200 --riemann=hll
    done
elif [[ "$mode" == luna ]]; then
    module purge
    module load gcc/13 cuda/12
    cd "$luna_dir"
    for p in sod shu-osher woodward-colella; do
        for n in ${sizes[$p]}; do
            for r in hllc hll; do
                "$luna_env/bin/python" -u "$repo/scripts/luna_1d_run.py" "$p" "$n" "$r" "$out/luna_${p}_${n}_$r.npz" 2>&1 | grep -E "^luna|Error"
            done
        done
    done
    for n in $large; do
        "$luna_env/bin/python" -u "$repo/scripts/luna_1d_run.py" sod "$n" hllc "$out/luna_sod_${n}_steps.npz" steps=200 2>&1 | grep -E "^luna|Error"
        "$luna_env/bin/python" -u "$repo/scripts/luna_1d_run.py" sod "$n" hll "$out/luna_sod_${n}_steps_hll.npz" steps=200 2>&1 | grep -E "^luna|Error"
    done
fi
