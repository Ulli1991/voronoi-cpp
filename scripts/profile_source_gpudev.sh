#!/bin/bash
#SBATCH --job-name=voronoi-source-profile
#SBATCH --partition=gpudev
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/source-profile-%j.out
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
# Reuse the exact lineinfo-enabled executable from the verified hardware capture.
exe="$PWD/build-roofline/31624143/build/reconstruction_profile"
test -x "$exe"
sha256sum "$exe" > "$profile_root/executable.sha256"
filter=(--target-processes all --kernel-name-base demangled --kernel-name 'regex:reconstruct_hydro<' --launch-skip 2 --launch-count 1)
check=python/check_ncu_capture.py
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
printf 'Source capture passed; correlate with hardware counters from job 31624143.\n' > "$profile_root/capture-complete.txt"
echo "Reconstruction profile artifacts: $profile_root"
