#!/bin/bash
# Build of the given targets and a run of the given tests off the login node; cuda as the third argument compiles with
# nvcc instead (no GPU needed to compile; tests then need a gpu1 node):
#   srun -p interactive -n1 -c8 --mem=16G --time=00:40:00 scripts/step2x_host_check.sh "<targets>" "<tests>" [cuda]
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
if [[ "${3:-}" == cuda ]]; then
    module load gcc/14 cuda/13.0 cmake
    dir=build-jobs/cuda-check
    cmake -S . -B $dir -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >/dev/null
else
    module load gcc/15 cmake
    dir=build-jobs/omp
    cmake -S . -B $dir -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos >/dev/null
fi
# shellcheck disable=SC2086
cmake --build $dir --target $1 -j"${SLURM_CPUS_PER_TASK:-8}" 2>&1 | grep -E "error|warning: " | head -20
echo "build exit ${PIPESTATUS[0]}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OMP_PROC_BIND=spread OMP_PLACES=cores
for t in $2; do
    echo "== $t"
    "$dir/$t" 2>&1 | tail -25
done
