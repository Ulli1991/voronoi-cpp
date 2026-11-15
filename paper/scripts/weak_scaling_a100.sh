#!/bin/bash
# The paper's weak-scaling sweep on one A100: warp rebuild, exact start and moving steps (update + update_cells) per
# input and size, one JSON line per timed run (library_bench). Rerun after each speed merge; submit_sweep.sh splits it.
#   sbatch -o paper/data/raw/weak-%j.jsonl -e paper/logs/weak-%j.err paper/scripts/weak_scaling_a100.sh <bin> <data>
#SBATCH -J paper_weak
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
bin="$1" data="$2"
export LIBBENCH_COMMIT="$(basename "$bin")"
mkdir -p "$data/points" "$data/clocks" "$data/tmp"
nvidia-smi -L >&2
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.applications.graphics,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader -l 1 >"$data/clocks/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null; rm -f "$data/tmp/"*"-$SLURM_JOB_ID.trj"' EXIT
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
