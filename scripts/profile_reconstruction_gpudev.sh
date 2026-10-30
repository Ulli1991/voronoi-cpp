#!/bin/bash
#SBATCH --job-name=voronoi-clip-profile
#SBATCH --partition=gpudev
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/clip-profile-%j.out
# Ask before submission. No monitor/service changes or automatic resubmission.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
profile_root="$PWD/build-roofline/${SLURM_JOB_ID:?}"
mkdir -p "$profile_root"
profile_phase=setup
preserve_failure() {
    profile_exit=$?
    if (( profile_exit != 0 )); then
        printf '%s\n' "$profile_phase" > "$profile_root/failed-stage.txt"
        /usr/bin/python3.11 - "$profile_root" <<'PY'
import pathlib
import re
import shutil
import sys
root = pathlib.Path(sys.argv[1])
for log in root.glob('*.log'):
    for name in set(re.findall(r'/tmp/nsight-compute-[\w-]+\.log', log.read_text(errors='replace'))):
        source = pathlib.Path(name)
        if source.is_file():
            shutil.copy2(source, root / ('driver-' + source.name))
PY
        echo "Profile incomplete at $profile_phase (exit $profile_exit); diagnostics: $profile_root"
    fi
}
trap preserve_failure EXIT
gcc_gpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/14.1.0
cuda_root=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1
export PATH="$cuda_root/bin:$gcc_gpu/bin:$PATH"
export LD_LIBRARY_PATH="$gcc_gpu/lib64:$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER="$gcc_gpu/bin/g++"
export OMP_NUM_THREADS=1
module load nsight_compute/2026
hostname
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,memory.total,memory.used --format=csv > "$profile_root/device.csv"
ncu --version
cmake -S . -B "$profile_root/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
    -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos
cmake --build "$profile_root/build" --target reconstruction_profile -j 4 > "$profile_root/build.log" 2>&1
exe="$profile_root/build/reconstruction_profile"
filter=(--target-processes all --kernel-name-base demangled --kernel-name 'regex:reconstruct_hydro<' --launch-skip 2 --launch-count 1)
check=python/check_ncu_capture.py
# Gate all further work on access to actual hardware counters for this kernel.
profile_phase=counter_access_preflight
probe_metrics=gpu__time_duration.sum,dram__bytes_read.sum
srun ncu "${filter[@]}" --metrics "$probe_metrics" --log-file "$profile_root/preflight.log" \
    --export "$profile_root/preflight" "$exe" 32 1 relative_early_scratch_sort
ncu --import "$profile_root/preflight.ncu-rep" --page raw --csv > "$profile_root/preflight.csv"
/usr/bin/python3.11 "$check" "$profile_root/preflight.csv" --required "$probe_metrics" > "$profile_root/preflight-verified.json"
profile_phase=uninstrumented_baseline
# Counts run in a SEPARATE untimed kernel after production timings/validation.
srun "$exe" 128 5 relative_early_scratch_sort "$profile_root/timings.csv" > "$profile_root/baseline-and-work.log"
profile_phase=hardware_counters
metrics=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__thread_inst_executed_per_inst_executed.ratio
srun ncu "${filter[@]}" --metrics "$metrics" --log-file "$profile_root/hardware.log" \
    --export "$profile_root/hardware-128" "$exe" 128 1 relative_early_scratch_sort
ncu --import "$profile_root/hardware-128.ncu-rep" --page raw --csv > "$profile_root/hardware-128.csv"
/usr/bin/python3.11 "$check" "$profile_root/hardware-128.csv" --required "$metrics" > "$profile_root/hardware-verified.json"
profile_phase=source_attribution
srun ncu "${filter[@]}" --section SourceCounters --section WarpStateStats --section InstructionStats \
    --warp-sampling-max-passes 1 --import-source on --source-folders "$PWD/src/voronoi_construct" \
    --log-file "$profile_root/source.log" --export "$profile_root/source-128" \
    "$exe" 128 1 relative_early_scratch_sort
ncu --import "$profile_root/source-128.ncu-rep" --page raw --csv > "$profile_root/source-128.csv"
/usr/bin/python3.11 "$check" "$profile_root/source-128.csv" --required smsp__inst_executed.sum > "$profile_root/source-verified.json"
ncu --import "$profile_root/source-128.ncu-rep" --page source --print-source cuda,sass > "$profile_root/source-128.txt"
/usr/bin/python3.11 - "$profile_root/source-128.txt" <<'PY'
import pathlib
import sys
text = pathlib.Path(sys.argv[1]).read_text()
if 'local_voronoi.hpp' not in text:
    raise SystemExit('Source attribution missing local_voronoi.hpp; inspect lineinfo/report')
PY
profile_phase=complete
printf 'Capture gates passed; analyze timings, hardware counters, source hotspots and work counts.\n' > "$profile_root/capture-complete.txt"
echo "Reconstruction profile artifacts: $profile_root"
