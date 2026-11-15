#!/bin/bash
# Mesh build on clustered and halo input (branch clustered-build) against main on one A100, every design in one job:
#   sbatch --export=ALL,DESIGNS="ref new" -o build-jobs/%x-%j.out scripts/clustered_build_a100.sh <mode> <n> [rounds]
# Binaries build-jobs/bin/voronoi_bench_<design>; VARIANTS="a b" runs `new` once per VORONOI_HARD value (0: off).
#SBATCH -J clustered_build
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --export=ALL
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
bin=${BIN:-build-jobs/bin}
mode="$1"
n="$2"
rounds="${3:-1}"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm,clocks.applications.graphics --format=csv,noheader
designs="${DESIGNS:-ref new}"
variants="${VARIANTS:-}"
points=/raven/u/uli/libbench-data/points
kinds="${KINDS:-jittered uniform clustered halo lattice wave bluenoise}"
# bluenoise is the library benchmark's file of that size.
input() { if [[ $1 == bluenoise ]]; then echo $points/bluenoise_$n.pts; else echo $1; fi; }
pairs() { for d in $designs; do if [[ $d == ref || -z $variants ]]; then echo "$d:-"; else for v in $variants; do echo "$d:$v"; done; fi; done; }
run() {  # run <design:variant> <args...>
    local d=${1%%:*} v=${1#*:}
    shift
    if [[ $v == - ]]; then env -u VORONOI_HARD $bin/voronoi_bench_$d "$@"; else VORONOI_HARD=$v $bin/voronoi_bench_$d "$@"; fi
}
for r in $(seq "$rounds"); do
    for kind in $kinds; do
        for p in $(pairs); do
            echo "== $mode $p $kind $n round $r"
            case "$mode" in
            rate) run $p rebuild "$n" "$(input $kind)" 5 --no-check | grep -E "^warp" ;;
            work) run $p rebuild "$n" "$(input $kind)" 1 | grep -vE "^#" ;;
            start) run $p step "$n" "$(input $kind)" 0.1 0 | grep -E "^start" ;;
            checksum) run $p step "$n" "$(input $kind)" 0.1 2 --checksum | grep -E "^(start|mesh|geometry_checksum)" ;;
            kernels) run $p rebuild "$n" "$(input $kind)" 3 --no-check --kernels | grep -E "^(warp|kernel)" ;;
            startk) run $p step "$n" "$(input $kind)" 0.1 0 --kernels | grep -E "^(start|kernel)" | head -30 ;;
            nsys)
                module load nsight_systems/2026 2>/dev/null
                out=build-jobs/nsys/${p//:/_}_${kind}_$n
                mkdir -p build-jobs/nsys
                d=${p%%:*} v=${p#*:}
                [[ $v != - ]] && export VORONOI_HARD=$v
                nsys profile -t cuda -f true -o $out $bin/voronoi_bench_$d rebuild "$n" "$(input $kind)" 3 --no-check \
                    >/dev/null 2>&1
                unset VORONOI_HARD
                nsys stats -q --report cuda_gpu_kern_sum --format csv $out.nsys-rep 2>/dev/null |
                    awk -F'","' 'NR > 1 { gsub(/"/, ""); printf "kern: %s total_ms=%.3f calls=%s\n", substr($9, 1, 90), $2 / 1e6, $3 }' |
                    head -12
                ;;
            measure) run $p rebuild "$n" "$(input $kind)" 1 --no-check --measure=1000000000 | grep -E "^measure" ;;
            esac
        done
    done
done
