#!/bin/bash
# Local repair (S2, tet_local) on one A100: main (ref) and the branch (new) in the same job.
#   sbatch -o build-cuda/gpu-run/s2-<part>-%j.txt scripts/local_repair_a100.sh <bin dir> <part> [n]
# <bin dir>/voronoi_bench_{ref,new} are copies of the binaries. Parts: verify, speed, crossover, often, smooth, surface.
#SBATCH -J local_repair
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
bin="$1" part="$2" n="${3:-2097152}" steps=${STEPS:-12}
echo "# bin=$bin host=$(hostname) job=${SLURM_JOB_ID:-0} n=$n"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics --format=csv,noheader | sed 's/^/# /'
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 1 >"$bin/clock-${SLURM_JOB_ID:-0}.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
run() {  # run <tag> <case label> <input> <spacings> <flags...>
    local tag="$1" label="$2"
    echo "== case $tag $label $3 $n $4 ${*:5}"
    timeout ${LIMIT:-300} "$bin/voronoi_bench_$tag" step "$n" "$3" "$4" "$steps" --no-start "${@:5}" |
        grep -E "^(#|motion|gpuflips|verify|refused|geometry_checksum)"
}
jumps() {  # the jump cases: label, spacings of the flow, flags
    echo "jump-2-1 jittered 0.1 --jump-share=$(python3 -c "print(2/$n)") --jump-spacings=1"
    echo "jump-2-2 jittered 0.1 --jump-share=$(python3 -c "print(2/$n)") --jump-spacings=2"
    echo "jump-2-4 jittered 0.1 --jump-share=$(python3 -c "print(2/$n)") --jump-spacings=4"
    for f in 1e-4 1e-3 1e-2; do echo "jump-$f-4 jittered 0.1 --jump-share=$f --jump-spacings=4"; done
    echo "flow-1 cosmicweb 1"
    echo "flow-0.5 clustered 0.5"
}
case "$part" in
verify)  # every step against a fresh exact rebuild, local repair forced and the default policy
    STEPS=${STEPS:-6}
    steps=$STEPS
    while read -r label kind d flags; do
        run new "$label-local" "$kind" "$d" $flags --fallback=local --verify
    done < <(jumps)
    ;;
speed)  # per-step stage times: main (cheapest), branch (local) and branch (rebuild)
    while read -r label kind d flags; do
        run ref "$label" "$kind" "$d" $flags
        run new "$label-local" "$kind" "$d" $flags --fallback=local
        run new "$label-cheapest" "$kind" "$d" $flags
    done < <(jumps)
    ;;
crossover)  # local against rebuild over the share of jumpers (4 spacings)
    for f in 1e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
        for policy in local rebuild; do
            run new "jump-$f-4-$policy" jittered 0.1 --jump-share=$f --jump-spacings=4 --fallback=$policy
        done
    done
    ;;
smooth)  # the default path: checksums and rates, seven inputs, three spacings
    STEPS=${STEPS:-10}
    steps=$STEPS
    for kind in jittered uniform clustered halo lattice wave bluenoise; do
        input=$kind
        [[ $kind == bluenoise ]] && input=/raven/u/uli/libbench-data/points/bluenoise_$n.pts
        for d in 0.03 0.1 0.3; do
            for tag in ref new ref new; do run "$tag" "smooth-$kind-$d" "$input" "$d" --checksum; done
        done
    done
    ;;
often)  # the inputs whose smooth flow sticks in a few steps: 30 steps under the default policy, every step verified
    STEPS=${STEPS:-30}
    steps=$STEPS
    run new "flow-1-cheapest" cosmicweb 1 --verify
    run new "flow-0.5-cheapest" clustered 0.5 --verify
    ;;
surface)  # the surface input (the start takes long; 64³ is enough)
    STEPS=${STEPS:-6}
    steps=$STEPS
    LIMIT=800 run new "flow-${D:-0.3}-surface-local" surface "${D:-0.3}" --fallback=local --verify
    ;;
esac
