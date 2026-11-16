#!/bin/bash
# Repeats of the cases job 32422876 left above 1 %, with its builds, from the repository root:
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/modes_rounding_repeat_a100.sh <job dir of 32422876>
# Exact start and steps of jittered and cosmicweb at 256³ and cosmicweb at 128³, main / gpu / same_bits, six rounds.
#SBATCH -J modes_repeat
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
dir=$(realpath "$1")
out="$(pwd)/build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out"
hostname
for round in 1 2 3 4 5 6; do
    for row in "16777216 jittered 0.1 6" "16777216 jittered 0.3 6" "16777216 cosmicweb 0.1 6" "2097152 cosmicweb 0.1 10"; do
        read -r n kind d steps <<<"$row"
        for b in main:main gpu:cuda same_bits:cuda; do
            name=${b%%:*} build=${b##*:}
            flags=()
            [[ $name == same_bits ]] && flags=(--same-bits)
            "$dir/$build/voronoi_bench" step "$n" "$kind" "$d" "$steps" "${flags[@]}" | grep -E "^(mesh|gpuflips):" |
                sed "s/^/round=$round build=$name kind=$kind n=$n d=$d /"
        done
    done
done >"$out/speed.log"
echo "speed lines: $(wc -l <"$out/speed.log")"
