#!/bin/bash
# voronoi-cpp (OpenMP) and voro++ (serial and multithreaded) in one job on one full CPU node, so their rows share the
# node: run_voronoi_cpp_cpu.sh, then run_voropp_cpu.sh. Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/cpu-pair-%j.jsonl -e <log> bench/libraries/run_cpu_pair.sh <build-omp> <data>
#SBATCH -J libbench_cpu_pair
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
here="$SLURM_SUBMIT_DIR/bench/libraries"
bash "$here/run_voronoi_cpp_cpu.sh" "$1" "$2" || echo "# voronoi-cpp CPU runs failed: exit $?" >&2
bash "$here/run_voropp_cpu.sh" "$2" || echo "# voro++ runs failed: exit $?" >&2
