#!/bin/bash
# tetgen in the library benchmark on one full general node (serial library: 1 thread): the Delaunay of the points
# padded with periodic images, and Delaunay + dual volumes (ours, images_dual.hpp), on every point file; 1 untimed run
# (peak memory, certificate, volumes) + 3 timed. Padding width per input in mean spacings (WIDTH_<input>), chosen from
# the certificate of a calibration run (see the note field of the records). Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/tetgen-cpu-%j.jsonl -e /raven/u/uli/libbench-data/logs/tetgen-cpu-%j.err \
#       bench/libraries/run_tetgen_cpu.sh [data dir] [sizes]
#SBATCH -J libbench_tetgen
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14
data="${1:-/raven/u/uli/libbench-data}"
sizes="${2:-2097152}"  # serial: 128^3 only
bin=/raven/u/uli/libbench-src/tetgen/install/bin
export OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=cores
declare -A width=([bluenoise]=${WIDTH_bluenoise:-3} [jittered]=${WIDTH_jittered:-3} [uniform]=${WIDTH_uniform:-4} [clustered]=${WIDTH_clustered:-8} [halo]=${WIDTH_halo:-8})
lscpu | grep -E 'Model name|^CPU\(s\)|MHz' >&2
for n in $sizes; do
    for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
        pts="$data/points/${kind}_${n}.pts"
        "$bin/tetgen_bench" "$pts" --variant=delaunay --width=${width[$kind]}
        "$bin/tetgen_bench" "$pts" --variant=faces --width=${width[$kind]} \
            --volumes="$data/volumes/tetgen-cpu_${kind}_${n}.f64"
    done
done
