#!/bin/bash
# The CUDA test programs under compute-sanitizer (memcheck, racecheck, initcheck or synccheck) on one A100.
#   sbatch -o <dir>/sanitize-%j.out scripts/sanitize_a100.sh <build-cuda> <tool> [name filter]
# One program per line of `programs` below (smallest sizes that fit 15 minutes); each runs under `timeout`.
#SBATCH -J sanitize
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
build="${1:-build-cuda}"
tool="${2:-memcheck}"
filter="${3:-.*}"
# name, seconds, program and arguments
programs=(
    "exact_predicates 60 exact_predicates_test"
    "clipping 120 clipping_test"
    "mesh 120 mesh_test"
    "start 120 start_test"
    "repair 120 repair_test"
    "voronoi_mesh 120 voronoi_mesh_test"
    "mesh1d 60 mesh1d_test"
    "mesh2d 120 mesh2d_test"
    "metamorphic 120 metamorphic_test"
    "mesh_cases.few 60 mesh_cases_test few."
    "mesh_cases.coincident 60 mesh_cases_test coincident."
    "mesh_cases.box 60 mesh_cases_test box."
    "mesh_cases.degenerate 60 mesh_cases_test degenerate."
    "mesh_cases.valence 60 mesh_cases_test valence."
    "mesh_cases.fallback 60 mesh_cases_test fallback."
    "mesh_cases.predicates 60 mesh_cases_test predicates."
)
flags=(--tool "$tool" --error-exitcode 99 --print-limit 10)
case "$tool" in
memcheck) flags+=(--leak-check full) ;;
racecheck) flags+=(--racecheck-report hazard) ;;
esac
export CUDA_LAUNCH_BLOCKING=0 OMP_NUM_THREADS=4
nvidia-smi -L
for entry in "${programs[@]}"; do
    read -r name seconds program args <<<"$entry"
    [[ $name =~ ^($filter) ]] || continue
    start=$SECONDS
    timeout -k 5 "$seconds" compute-sanitizer "${flags[@]}" "$build/$program" $args > "/tmp/sanitize.$$" 2>&1
    status=$?
    printf "program=%s tool=%s exit=%s seconds=%s errors=%s\n" "$name" "$tool" "$status" "$((SECONDS - start))" \
        "$(grep -c '^========= ' "/tmp/sanitize.$$")"
    grep -E '^=========' "/tmp/sanitize.$$" | head -40 | cut -c1-300
    grep -E '^  FAIL' "/tmp/sanitize.$$" | head -5 | cut -c1-300
done
rm -f "/tmp/sanitize.$$"
