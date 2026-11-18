#!/bin/bash
# Re-runs fuzz seeds with the mesh_fuzz_test of several builds, one line each:
#   srun -p interactive -n1 -c8 --mem=24G --time=01:00:00 scripts/step2x_fuzz_seeds.sh "<build dirs>" <seed>...
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
builds="$1"
shift
for seed in "$@"; do
    for b in $builds; do
        echo "$b $(timeout 300 "$b/mesh_fuzz_test" one "$seed" 2>&1 | grep -E 'exact|refused|valid|bits|rough|WRONG|CRASH|HANG' | tail -1 | cut -c1-160)"
    done
done
