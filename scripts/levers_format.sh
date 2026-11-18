#!/bin/bash
# clang-format 18 (the CI's) on the given files of the levers tree, in place, then the comment check:
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_format.sh <file>...
#SBATCH -J lev_format
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
cd /raven/u/uli/voronoi-cpp-levers || exit 1
/mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8/bin/clang-format -i "$@" && echo "formatted: $*"
python3 scripts/check_comments.py && echo "comments: ok"
