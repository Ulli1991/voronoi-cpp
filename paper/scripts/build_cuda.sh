#!/bin/bash
# CUDA builds of library_bench, voronoi_bench and paper_trajectory for the sweep, copied to <bin>/<commit>/ so a rebuild
# never races a queued job:  sbatch -o paper/logs/build-%j.out paper/scripts/build_cuda.sh [bin dir]
#SBATCH -J paper_build
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=00:40:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
root="$(git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)"
commit="$(git -C "$root" rev-parse --short HEAD)$(git -C "$root" diff --quiet HEAD -- src bench || echo +local)"
bin="${1:-/ptmp/uli/voronoi-paper/bin}/$commit"
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper
    -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos -DBUILD_TESTING=OFF)
cmake -S "$root" -B "$root/build-paper-cuda" "${flags[@]}" >/dev/null
cmake --build "$root/build-paper-cuda" -j 18 --target library_bench voronoi_bench
cmake -S "$root/paper/bench" -B "$root/build-paper-cuda/paper" "${flags[@]}" >/dev/null
cmake --build "$root/build-paper-cuda/paper" -j 18
mkdir -p "$bin"
cp "$root"/build-paper-cuda/{library_bench,voronoi_bench} "$root/build-paper-cuda/paper/paper_trajectory" "$bin/"
echo "binaries in $bin"
