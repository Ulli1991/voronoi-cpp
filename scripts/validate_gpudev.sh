#!/bin/bash
#SBATCH --job-name=voronoi-table-check
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/job-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
validation_root="$PWD/build-table-validation"
gcc_cpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/15.1.0
gcc_gpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/14.1.0
cuda_root=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1
export PATH="$cuda_root/bin:$gcc_gpu/bin:$PATH"
export LD_LIBRARY_PATH="$gcc_cpu/lib64:$gcc_gpu/lib64:$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER="$gcc_gpu/bin/g++"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

hostname
nvidia-smi --query-gpu=name,driver_version --format=csv
cmake -S . -B "$validation_root/cpu" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$gcc_cpu/bin/g++" \
    -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos
cmake --build "$validation_root/cpu" --target table_mesh_check -j 4
cmake -S . -B "$validation_root/gpu" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
    -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos
cmake --build "$validation_root/gpu" --target table_mesh_check -j 4
srun --ntasks=1 "$validation_root/cpu/table_mesh_check" "$validation_root/cpu-faces.csv"
srun --ntasks=1 "$validation_root/gpu/table_mesh_check" "$validation_root/gpu-faces.csv"
python3 - "$validation_root/cpu-faces.csv" "$validation_root/gpu-faces.csv" <<'PY'
import csv
import math
import sys

def load(path):
    faces = {}
    with open(path) as stream:
        for row in csv.reader(stream):
            key = (int(row[0]), int(row[1]))
            if key in faces:
                raise RuntimeError(f"duplicate face {key} in {path}")
            faces[key] = tuple(map(float, row[2:]))
    if not faces:
        raise RuntimeError(f"no faces in {path}")
    return faces

cpu, gpu = map(load, sys.argv[1:])
if cpu.keys() != gpu.keys():
    raise RuntimeError("CPU/GPU face neighbor IDs differ")
max_error = 0.0
for key in cpu:
    for a, b in zip(cpu[key], gpu[key]):
        if not math.isfinite(a) or not math.isfinite(b):
            raise RuntimeError(f"nonfinite face area at {key}")
        max_error = max(max_error, abs(a-b))
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"CPU/GPU area mismatch at {key}: {a} vs {b}")
print(f"CPU/GPU comparison PASS faces={len(cpu)} max_area_component_error={max_error:.6g}")
PY
