#!/bin/bash
# Checks of voronoi-cpp on the application inputs (cosmic web, surface, fcc solid): the warp rebuild with check_cells
# (closure, volume sum, faces seen from both sides, equidistant face centroids) and 10 moving steps of 0.03 spacings with --verify.
#   sbatch -o <log> scripts/inputs_check_a100.sh <build dir with voronoi_bench>
#SBATCH -J inputs_check
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
build="$1"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
for case in "cosmicweb 2097152" "surface 2097152" "fccsolid 1048576" "cosmicweb 16777216" "surface 16777216" \
    "fccsolid 8388608"; do
    set -- $case
    "$build/voronoi_bench" rebuild "$2" "$1" 3 --check || echo "# FAILED rebuild check $1 $2: exit $?"
done
# The verify compares every step with a fresh build (host side), so at 64³ (fcc: 32³ unit cells), as bench_a100.sh.
for case in "cosmicweb 262144" "surface 262144" "fccsolid 131072"; do
    set -- $case
    "$build/voronoi_bench" step "$2" "$1" 0.03 10 --verify | grep -E "^start|verify|FAIL|refus" \
        || echo "# FAILED step verify $1 $2"
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
