#!/bin/bash
# clang-format 18 (the CI's) on the lines the levers tree changed since <base> (working tree included), in place, then
# the comment check: lines older than .clang-format stay as they are, as scripts/check_format.py checks.
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_format.sh <base>
#SBATCH -J lev_format
#SBATCH -p interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
cd /raven/u/uli/voronoi-cpp-levers || exit 1
base="${1:-1741a1a}"
format=/mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8/bin/clang-format
for file in $(git diff --name-only "$base" -- '*.hpp' '*.cpp'); do
    lines=()
    while read -r start count; do
        ((count == 0)) && continue
        lines+=(--lines="$start:$((start + count - 1))")
    done < <(git diff -U0 "$base" -- "$file" | sed -nE 's/^@@ -[^ ]+ \+([0-9]+)(,([0-9]+))? @@.*/\1 \3/p' |
        awk '{print $1, ($2 == "" ? 1 : $2)}')
    ((${#lines[@]})) && "$format" -i --style=file "${lines[@]}" "$file" && echo "formatted: $file (${#lines[@]} ranges)"
done
python3 scripts/check_comments.py && echo "comments: ok"
