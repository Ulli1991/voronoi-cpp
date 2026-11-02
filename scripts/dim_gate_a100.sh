#!/bin/bash
# The 3D gate of a dimension stage (research_notes/dimension_design_2026-11-02.md,
# 6.3) on one A100: two CUDA builds, MAIN and FIX (paths), interleaved
# command by command (timed commands twice, MAIN FIX MAIN FIX):
#   ctest of FIX; warp rebuild of 2.1M points on six inputs (check_cells,
#   times); `step 262144 <kind> 0.3 3 --verify` on six inputs; steps at
#   0.03 / 0.1 / 0.3 spacings at 2.1M (start, step phases, update_cells,
#   cells_checksum, --memory); the moving hydro step at 128^3 (40 steps,
#   --memory); two moving runs per build at 128^3 (STEPS steps, default 200)
#   writing their final state, compared bit for bit afterwards.
# Submit from the repository root, logs on a shared file system:
#   MAIN=<build> FIX=<build> sbatch -o <logs>/gate-%j.out scripts/dim_gate_a100.sh <out>
#SBATCH -J dim_gate
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
out="$(realpath -m "${1:-gate-runs}")"
mkdir -p "$out"
nvidia-smi -L
ctest --test-dir "$FIX" --output-on-failure | tail -3
for kind in jittered uniform clustered halo lattice wave; do
    for variant in MAIN FIX; do
        echo "== $variant rebuild $kind"
        "${!variant}/voronoi_bench" rebuild 2097152 "$kind" warp 3 | grep -vE "^  "
    done
done
for kind in jittered uniform clustered halo lattice wave; do
    for variant in MAIN FIX; do
        echo "== $variant verify $kind"
        "${!variant}/voronoi_bench" step 262144 "$kind" 0.3 3 --verify | grep -E "^start|verify|checksum"
    done
done
for d in 0.03 0.1 0.3; do
    for variant in MAIN FIX MAIN FIX; do
        echo "== $variant step 2097152 jittered $d"
        "${!variant}/voronoi_bench" step 2097152 jittered "$d" 10 --memory | grep -vE "^  "
    done
done
for variant in MAIN FIX MAIN FIX; do
    echo "== $variant hydro 128^3 jittered moving --steps=40"
    "${!variant}/hydro_sedov" 128 jittered --moving --steps=40 --memory | grep -vE "^  "
done
for variant in MAIN FIX; do
    for run in 1 2; do
        echo "== $variant hydro 128^3 moving run $run, final state"
        "${!variant}/hydro_sedov" 128 jittered --moving --snapshots=1 --steps="${STEPS:-200}" \
            --out="$out/repro_${variant}_$run" | grep -E "step|final|updates" | tail -3
    done
done
# The last snapshot of each run: hashes of its U, x and volume.
for dir in "$out"/repro_*; do
    f="$(ls "$dir"/snapdir_*/*.hdf5 | tail -1)"
    sums=""
    for set in U x volume; do
        h5dump -d "/cells/$set" -b LE -o "$dir/$set.bin" "$f" > /dev/null
        sums+=" $set=$(sha256sum < "$dir/$set.bin" | cut -c1-16)"
    done
    echo "hash: $(basename "$dir")$sums"
done
