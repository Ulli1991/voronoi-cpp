#!/bin/bash
# Hydro speed and precision options on one A100 (research_notes/hydro_speed_2026-11-04.md).
# MAIN and FIX are two CUDA build directories (main and this branch), passed
# with sbatch --export=ALL,MAIN=...,FIX=... (Raven's sbatch does not export
# the environment otherwise). Submit from the repository root:
#   MAIN=<build> FIX=<build> sbatch -p gpudev -o <logs>/speed-%j.out scripts/hydro_speed_a100.sh <out> <mode>
# Modes:
#   speed128   MAIN and FIX interleaved (MAIN FIX MAIN FIX), 128^3, static and
#              moving, 40 steps, --memory; then every precision option of FIX
#              (static and moving, 40 steps, twice)
#   speed256   the same at 256^3: MAIN and FIX once each, then the options
#              (PRECISIONS, default "mixed mixed16 low")
#   accuracy   FIX, every option (PRECISIONS, default all): the whole Sedov at
#              128^3, static and moving, final snapshot in <out>/<option>_<mesh>
#   stream     FIX, every option: uniform streams (--no-blast) at 10 and 500
#              km/s, 40 static and 40 moving steps
#   repro      FIX, every option: two identical runs (static 100 steps, moving
#              100 steps), checksums of the final state, positions, volumes
#   bits       MAIN and FIX: final-state hashes of five paths (50 steps, 128^3)
#   ctest      ctest of FIX
#   ncu        Nsight Compute of FIX's hydro kernels, 128^3, static and moving
#              (PRECISIONS, default "default")
#SBATCH -J hydro_speed
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
out="$(realpath -m "${1:-speed-runs}")"
mode="${2:-speed128}"
mkdir -p "$out"
nvidia-smi -L
all="default weight32 weight64 flux32 grad32 grad16 geo32 riemann32 mixed mixed16 low lowweights"

speed() {  # speed <build> <side> [flags]: the phase lines of a 40-step run
    local build="$1" side="$2"
    shift 2
    "$build/hydro_sedov" "$side" jittered --steps=40 --memory "$@" | grep -E "^(mesh|run|phases|throughput|memory|conservation|fallbacks)"
}

case "$mode" in
speed128 | speed256)
    side=128
    options="${PRECISIONS:-$all}"
    rounds="1 2"
    if [[ "$mode" == speed256 ]]; then
        side=256
        options="${PRECISIONS:-mixed mixed16 low}"
        rounds="1"
    fi
    # Particle order: Morton (default) and as generated (--input-order: lattice rows).
    for round in $rounds; do
        for mesh in "" "--moving"; do
            for order in "" "--input-order"; do
                for variant in MAIN FIX; do
                    echo "== $variant ${side}^3 ${mesh:-static} ${order:-morton} round $round"
                    speed "${!variant}" $side $mesh $order
                done
            done
        done
    done
    for round in $rounds; do
        for p in $options; do
            for mesh in "" "--moving"; do
                for loop in ${UPDATES:-face}; do
                    echo "== FIX ${side}^3 ${mesh:-static} precision=$p update=$loop round $round"
                    speed "$FIX" $side $mesh --precision=$p --update=$loop
                done
            done
        done
    done
    ;;
builds)
    # Experiments: the builds in BUILDS (paths), interleaved, 128^3, static
    # and moving, the options in PRECISIONS (default double), twice.
    for round in 1 2; do
        for build in $BUILDS; do
            for p in ${PRECISIONS:-default}; do
                for mesh in "" "--moving"; do
                    echo "== $(basename "$build") 128^3 ${mesh:-static} precision=$p round $round"
                    speed "$build" 128 $mesh --precision=$p
                done
            done
        done
    done
    ;;
accuracy)
    for p in ${PRECISIONS:-$all}; do
        for mesh in static moving; do
            flags=""
            [[ "$mesh" == moving ]] && flags="--moving"
            echo "== FIX Sedov 128^3 $mesh precision=$p"
            "$FIX/hydro_sedov" 128 jittered $flags --precision=$p --snapshots=1 --out="$out/${p}_$mesh" |
                grep -E "^(run|fallbacks|phases|throughput|conservation|checksum|snapshot: index=1)"
        done
    done
    ;;
stream)
    for p in ${PRECISIONS:-$all}; do
        for bulk in 10 500; do
            for mesh in "" "--moving"; do
                echo "== FIX stream 64^3 bulk=$bulk ${mesh:-static} precision=$p"
                "$FIX/hydro_sedov" 64 jittered --no-blast --bulk=$bulk --steps=40 $mesh --precision=$p |
                    grep -E "^(run|conservation|snapshot)"
            done
        done
    done
    ;;
repro)
    for p in ${PRECISIONS:-$all}; do
        for mesh in "" "--moving"; do
            for run in 1 2; do
                echo "== FIX repro 128^3 ${mesh:-static} precision=$p run $run"
                "$FIX/hydro_sedov" 128 jittered --steps=100 $mesh --precision=$p | grep -E "^checksum"
            done
        done
    done
    ;;
bits)
    # MAIN and FIX, the paths of scripts/dim_gate_a100.sh, 50 steps at
    # 128^3: hashes of the final U, x and volume (scripts/snapshot_hash.sh).
    for variant in MAIN FIX; do
        for flags in "" "--moving" "--order=0" "--update=cell --moving" "--riemann=hll --moving"; do
            tag="$(echo "$flags" | tr -d ' -=')"
            "${!variant}/hydro_sedov" 128 jittered $flags --snapshots=1 --steps=50 --out="$out/bits_${tag:-static}_$variant" > /dev/null
        done
    done
    bash scripts/snapshot_hash.sh "$out"/bits_* | sort
    ;;
ctest)
    ctest --test-dir "$FIX" --output-on-failure | tail -5
    ;;
ncu)
    # As the precision study (branch precision-study, scripts/precision_a100.sh:157-171):
    # Raven's DCGM holds the counters unless paused.
    dcgmi profile --pause
    for p in ${PRECISIONS:-default}; do
        for mesh in "" "--moving"; do
            echo "== ncu FIX 128^3 ${mesh:-static} precision=$p ${NCU_FLAGS:-}"
            /mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu --target-processes all \
                --print-kernel-base demangled --launch-skip 60 --launch-count 120 \
                --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed \
                --csv --page raw "$FIX/hydro_sedov" 128 jittered $mesh --steps=20 --precision=$p ${NCU_FLAGS:-} \
                > "$out/ncu_${p}_${mesh:-static}${NCU_TAG:-}.csv" 2> "$out/ncu_${p}_${mesh:-static}${NCU_TAG:-}.err"
            tail -2 "$out/ncu_${p}_${mesh:-static}${NCU_TAG:-}.err"
        done
    done
    dcgmi profile --resume
    ;;
esac
