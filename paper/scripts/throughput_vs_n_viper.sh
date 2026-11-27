#!/bin/bash
# The sweep of throughput_vs_n_a100.sh on one Viper APU (MPCDF): the MI300A (a HIP build) or its 24 host cores (an OpenMP
# build), the same inputs, paths, repeats and record format. The build decides the backend; the job names the device.
#   sbatch -o paper/data/raw/sweep-%j.jsonl -e paper/logs/sweep-%j.err --export=ALL,ENV=<env script>[,SIZES=..,KINDS=..] \
#       paper/scripts/throughput_vs_n_viper.sh <bin> <data> <commit> <device label>
#SBATCH -J paper_sweep
#SBATCH -p apudev,apu1
#SBATCH --gres=gpu:1
#SBATCH -N 1 --ntasks=1 --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=00:15:00
set -uo pipefail
source "${ENV:?the environment script of the build (compiler, Kokkos, ROCm)}"
bin="$1" data="$2"
export LIBBENCH_COMMIT="$3" LIBBENCH_DEVICE="$4"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}" OMP_PROC_BIND=spread OMP_PLACES=cores
root="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$data/points" "$data/tmp"
checks="$root/paper/data/raw/check-$SLURM_JOB_ID.txt"
{ hostname; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket'; rocm-smi --showproductname --showclocks 2>/dev/null | grep -E 'GPU\[|sclk|Card'; } >&2
trap 'rm -f "$data/tmp/"*"-$SLURM_JOB_ID.trj"' EXIT
points() {  # points <kind> <n>: the point file, made once (blue noise: 100 Lloyd iterations from uniform)
    local f="$data/points/$1_$2.pts"
    if [[ ! -s "$f" ]]; then
        if [[ "$1" == bluenoise ]]; then
            [[ -s "$data/points/uniform_$2.pts" ]] || "$bin/library_bench" points "$2" uniform "$data/points/uniform_$2.pts" >&2
            "$bin/library_bench" lloyd "$data/points/uniform_$2.pts" 100 "$f" >&2
        else
            "$bin/library_bench" points "$2" "$1" "$f" >&2
        fi
    fi
    echo "$f"
}
for n in ${SIZES:-32768 262144 2097152}; do
    for kind in ${KINDS:-jittered bluenoise uniform clustered halo lattice wave}; do
        f=$(points "$kind" "$n")
        if [[ "$kind" != bluenoise && -z "${NOCHECK:-}" ]]; then  # cell checks and faces per cell of a rebuild
            echo "== check $kind $n" >>"$checks"
            "$bin/voronoi_bench" rebuild "$n" "$kind" 3 >>"$checks" 2>&1 || echo "# failed: check $kind $n" >&2
        fi
        for path in ${PATHS:-warp exact}; do
            [[ "$path" == none ]] && continue
            "$bin/library_bench" build "$f" --path=$path --repeats=${REPEATS:-5} || echo "# failed: build $path $f" >&2
        done
        for d in ${SPACINGS:-0.03 0.1 0.3}; do
            t="$data/tmp/${kind}_${n}_$d-$SLURM_JOB_ID.trj"
            "$bin/paper_trajectory" "$f" "$d" "${STEPS:-10}" "$t" >&2 &&
                "$bin/library_bench" step "$t" | sed "s/\"phase\":\"step\"/\"phase\":\"step\",\"spacings\":$d/" ||
                echo "# failed: step $d $f" >&2
            rm -f "$t"
        done
    done
done
echo "# end $(date +%T) after ${SECONDS}s" >&2
