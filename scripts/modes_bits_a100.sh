#!/bin/bash
# After modes_a100.sh (job 32378021): where the CPU and GPU bits part, on one gpu1 node, from the repository root:
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/modes_bits_a100.sh <both> <nofma> <main>
# <both>: build on the CUDA + OpenMP Kokkos (nvcc's default --fmad=true); <nofma>: the same with --fmad=false;
# <main>: the CUDA-only build. All built on the login node beforehand. PARTS="1 2" (default "1 2 3 4") runs only those.
# 1. ctest of <main> and <both> on the A100 (modes_test: cpu, gpu and hybrid), and modes_test of <nofma>.
# 2. backends_probe in <both> and <nofma>: OpenMP against CUDA (bits, volume errors against a long-double reference),
#    and the clipping's kernels apart from the spaces.
# 3. DFMA instructions in the device code of each.
# 4. Speed, interleaved: <both> against <main> (the cost of the second backend), <nofma> against <both> (the cost of
#    the unfused device arithmetic): rebuild, exact start and steps.
#SBATCH -J modes_bits
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=03:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
both=$(realpath "$1") nofma=$(realpath "$2") main=$(realpath "$3")
out="$(pwd)/build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm --format=csv,noheader
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

parts=" ${PARTS:-1 2 3 4} "
if [[ $parts == *" 1 "* ]]; then
echo "== 1. A100 ctest"
for b in main both; do
    dir=${!b}
    ctest --test-dir "$dir" --output-on-failure >"$out/ctest_$b.log" 2>&1
    echo "ctest $b: $(grep -E 'tests passed|tests failed' "$out/ctest_$b.log")"
    grep -E "^ *[0-9]+ - " "$out/ctest_$b.log"
done
for b in both nofma; do
    echo "-- modes_test $b"
    "${!b}/modes_test" >"$out/modes_$b.log" 2>&1
    echo "modes_test $b exit $?: $(grep -c '^ *ok' "$out/modes_$b.log") ok, $(grep -c '^ *FAIL' "$out/modes_$b.log") FAIL"
    grep -E "FAIL|gpu|hybrid" "$out/modes_$b.log" | head -40
done

fi
if [[ $parts == *" 2 "* ]]; then
echo "== 2. backends_probe"
for b in both nofma; do
    for kind in ${KINDS:-uniform clustered cosmicweb}; do
        echo "-- $b $kind 128"
        timeout 1500 "${!b}/backends_probe" 2097152 "$kind" 0.1 2 0 | grep -E "^(clip|same|backend|volume_)"
    done
done

fi
if [[ $parts == *" 3 "* ]]; then
echo "== 3. DFMA in the device code"
for b in both nofma main; do
    echo "$b backends_probe/voronoi_bench: DFMA $(cuobjdump -sass "${!b}/voronoi_bench" | grep -c DFMA)," \
        "DMUL $(cuobjdump -sass "${!b}/voronoi_bench" | grep -c DMUL), DADD $(cuobjdump -sass "${!b}/voronoi_bench" | grep -c DADD)"
done
echo "host vfmadd in both/voronoi_bench: $(objdump -d "$both/voronoi_bench" | grep -c vfmadd)"

fi
if [[ $parts == *" 4 "* ]]; then
echo "== 4. speed, interleaved"
for round in 1 2 3; do
    for kind in jittered uniform clustered halo cosmicweb; do
        for b in main both nofma; do
            "${!b}/voronoi_bench" rebuild 2097152 "$kind" 5 --no-check | grep "^warp:" |
                sed "s/^/round=$round build=$b kind=$kind n=2097152 /"
        done
        for d in 0.03 0.3; do
            for b in main both nofma; do
                "${!b}/voronoi_bench" step 2097152 "$kind" "$d" 10 | grep -E "^(mesh|gpuflips):" |
                    sed "s/^/round=$round build=$b kind=$kind n=2097152 d=$d /"
            done
        done
    done
    for b in main both nofma; do
        "${!b}/voronoi_bench" rebuild 16777216 jittered 3 --no-check | grep "^warp:" |
            sed "s/^/round=$round build=$b kind=jittered n=16777216 /"
        "${!b}/voronoi_bench" step 16777216 jittered 0.03 6 | grep -E "^(mesh|gpuflips):" |
            sed "s/^/round=$round build=$b kind=jittered n=16777216 d=0.03 /"
    done
done >"$out/speed.log"
echo "speed lines: $(wc -l <"$out/speed.log")"
fi
