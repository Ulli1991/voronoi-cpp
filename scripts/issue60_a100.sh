#!/bin/bash
# Issue #60 on one A100: two builds (ref, new) in the same job, interleaved.
#   sbatch -o build-jobs/%x-%j.out scripts/issue60_a100.sh <mode> <n> [rounds]
# Binaries build-jobs/bin/vb_{ref,new} and lb_{ref,new}; modes: bits (checksums of the start, the mesh, 3 moving steps
# and the warp rebuild), rate (warp rebuild construct_ms and exact start start_ms), surface (rebuild with the cell check
# and volumes of the library paths for the oracle), library (warp and exact paths of the library bench on blue noise).
#SBATCH -J issue60
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
bin=${BIN:-build-jobs/bin}
data=${DATA:-/raven/u/uli/libbench-data}
mode="$1"
n="$2"
rounds="${3:-1}"
kinds="${KINDS:-jittered uniform clustered halo lattice wave cosmicweb fccsolid}"
hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
for r in $(seq "$rounds"); do
    for kind in $kinds; do
        for d in ref new; do
            echo "== $mode $d $kind $n round $r"
            case "$mode" in
            bits)
                $bin/vb_$d step "$n" "$kind" 0.03 3 --checksum | grep -E "^start|^mesh|geometry_checksum"
                $bin/vb_$d rebuild "$n" "$kind" 1 | grep -E "clip_checksum|certified"
                ;;
            rate)
                $bin/vb_$d rebuild "$n" "$kind" 5 --no-check | grep -E "^warp"
                $bin/vb_$d step "$n" "$kind" 0.03 0 | grep -E "^start"
                ;;
            surface)
                $bin/vb_$d rebuild "$n" surface "$rounds" | grep -E "^warp|certified|work1"
                ;;
            library)
                for path in warp exact; do
                    $bin/lb_$d build "$data/points/${kind}_${n}.pts" --path=$path --repeats=3 \
                        --volumes="build-jobs/volumes/${d}-${path}_${kind}_${n}.f64" | grep -oE '"(phase|Mcells_s)":"?[0-9a-z.e+-]*' | tr '\n' ' '
                    echo
                done
                ;;
            esac
        done
    done
    [ "$mode" = surface ] && break
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
