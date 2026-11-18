#!/bin/bash
# Repeats host test cases of a CI build (sanitizers on) to see how often a run-to-run failure shows:
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_host_repeat.sh <ci build dir> <repeats>
#SBATCH -J lev_repeat
#SBATCH -p interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1
dir="$1" repeats="${2:-10}"
err="$dir/repeat-${SLURM_JOB_ID:-0}.err"
for ((r = 1; r <= repeats; ++r)); do
    "$dir/mesh_cases_test" large.after_refusal.step.host >/dev/null 2>"$err"
    echo "large.after_refusal.step.host $r exit $? $(grep -m1 'runtime error' "$err")"
done
for ((r = 1; r <= 3; ++r)); do
    "$dir/modes_test" >/dev/null 2>"$err"
    echo "modes $r exit $? $(grep -m1 'runtime error' "$err")"
done
