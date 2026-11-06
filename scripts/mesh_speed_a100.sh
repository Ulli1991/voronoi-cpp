#!/bin/bash
# Mesh-step speed on one A100 (research_notes/mesh_speed_2026-11-04.md); BUILDS lists CUDA builds, modes are the cases below.
#   sbatch --export=ALL,BUILDS="a b" -p gpu1,gpudev -o <logs>/mesh-%j.out scripts/mesh_speed_a100.sh <out> <mode>
#SBATCH -J mesh_speed
#SBATCH -p gpu1,gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
out="$(realpath -m "${1:-mesh-runs}")"
mode="${2:-speed128}"
mkdir -p "$out"
hostname
nvidia-smi -L
nvidia-smi --query-gpu=clocks.max.sm,clocks.current.sm --format=csv,noheader

steps() {  # steps <n> <rounds> <steps>
    for round in $(seq 1 "$2"); do
        for d in ${SPACINGS:-0.03 0.1 0.3}; do
            for b in $BUILDS; do
                echo "== $(basename "$b") step $1 jittered $d round $round"
                "$b/voronoi_bench" step "$1" jittered "$d" "$3" --memory | grep -E "^(gpuflips|memory: phase=steps|mesh:)"
            done
        done
    done
}
hydro() {  # hydro <side> <rounds>
    for round in $(seq 1 "$2"); do
        for b in $BUILDS; do
            echo "== $(basename "$b") hydro $1^3 moving round $round"
            "$b/hydro_sedov" "$1" jittered --moving --steps=40 --memory | grep -E "^(mesh|run|phases|throughput|memory|fallbacks)"
        done
    done
}

case "$mode" in
kernels)
    for b in $BUILDS; do
        for d in ${SPACINGS:-0.03 0.1 0.3}; do
            echo "== $(basename "$b") kernels 2097152 jittered $d"
            CUDA_LAUNCH_BLOCKING=1 "$b/voronoi_bench" step 2097152 jittered "$d" 10 --kernels | grep -E "^(gpuflips|kernel)"
        done
    done
    for b in $BUILDS; do
        for d in ${SPACINGS:-0.03 0.1 0.3}; do
            echo "== $(basename "$b") touched 2097152 jittered $d"
            "$b/voronoi_bench" step 2097152 jittered "$d" 6 --touched | grep -E "^(gpuflips|touched)"
        done
    done
    ;;
speed128)
    steps 2097152 "${ROUNDS:-3}" 10
    hydro 128 "${HYDRO_ROUNDS:-2}"
    ;;
speed256)
    steps 16777216 1 10
    hydro 256 1
    ;;
bits)
    for b in $BUILDS; do
        for flags in "--moving" "" "--order=0" "--update=cell --moving" "--riemann=hll --moving"; do
            tag="$(echo "$flags" | tr -d ' -=')"
            "$b/hydro_sedov" 128 jittered $flags --snapshots=1 --steps=50 --out="$out/$(basename "$b")_${tag:-static}" > /dev/null
        done
    done
    bash scripts/snapshot_hash.sh "$out"/*_* | sort -t_ -k2
    ;;
verify)
    for b in $BUILDS; do
        for kind in jittered uniform clustered halo lattice wave; do
            echo "== $(basename "$b") verify $kind"
            "$b/voronoi_bench" step 262144 "$kind" 0.3 3 --verify | grep -E "verify"
        done
    done
    ;;
mem256)
    # Moving 256^3: per face in all-double and mixed, per cell; 40 steps, --memory.
    for b in $BUILDS; do
        for flags in "" "--precision=mixed" "--update=cell" ${MEM_FLAGS:-}; do
            echo "== $(basename "$b") hydro 256^3 moving $flags"
            "$b/hydro_sedov" 256 jittered --moving --steps=40 --memory $flags | grep -E "^(mesh|run|phases|throughput|memory)|Kokkos ERROR" | cut -c1-250
        done
    done
    ;;
tend256)
    for b in $BUILDS; do
        echo "== $(basename "$b") hydro 256^3 moving to t_end ${TEND_FLAGS:-}"
        "$b/hydro_sedov" 256 jittered --moving --memory ${TEND_FLAGS:-} | grep -E "^(mesh|run|phases|throughput|memory|mesh_updates|conservation)|Kokkos ERROR" | cut -c1-250
    done
    ;;
checksums)
    for b in $BUILDS; do
        for d in 0.03 0.1 0.3; do
            echo "== $(basename "$b") checksums $d"
            "$b/voronoi_bench" step 2097152 jittered "$d" 5 --checksum | grep -E "^geometry_checksum"
        done
    done
    ;;
check)
    for b in $BUILDS; do
        "$b/repair_test" | grep -vE "^ok|^  ok" | tail -8
        "$b/voronoi_mesh_test" | grep -vE "^ok|^  ok" | tail -8
        for kind in jittered uniform clustered wave; do
            echo "== $(basename "$b") verify $kind"
            "$b/voronoi_bench" step 262144 "$kind" 0.3 3 --verify | grep -E "verify" | cut -c1-200
            "$b/voronoi_bench" step 262144 "$kind" 0.03 3 --verify | grep -E "verify" | cut -c1-200
        done
    done
    ;;
ctest)
    for b in $BUILDS; do
        ctest --test-dir "$b" --output-on-failure | tail -4
    done
    ;;
esac
