#!/bin/bash
# Host (OpenMP) build of the given targets and a run of the given tests, for a quick check off the login node:
#   srun -p interactive -n1 -c8 --mem=16G --time=00:40:00 scripts/step2x_host_check.sh "<targets>" "<tests>"
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15 cmake
cmake -S . -B build-jobs/omp -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos >/dev/null
# shellcheck disable=SC2086
cmake --build build-jobs/omp --target $1 -j"${SLURM_CPUS_PER_TASK:-8}" 2>&1 | grep -E "error|warning: " | head -20
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OMP_PROC_BIND=spread OMP_PLACES=cores
for t in $2; do
    echo "== $t"
    "build-jobs/omp/$t" 2>&1 | tail -25
done
