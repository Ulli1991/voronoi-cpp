#!/bin/bash
# Paragram for the library benchmark: a Python venv with PyTorch (CUDA 12.8 wheels), the repo at the pinned commit
# with its cuBQL submodule, and the two lazy CUDA builds (Paragram's own extension, as its _backend.py compiles it on
# first import, and the driver's volume kernel in paragram.py) into /raven/u/uli/libbench-src/paragram/torch_ext.
#   fetch (login node, downloads only):  bench/libraries/build_paragram.sh fetch
#   build (Slurm, A100 for sm_80):       sbatch -o /raven/u/uli/libbench-data/logs/build-paragram-%j.out \
#                                               bench/libraries/build_paragram.sh
#SBATCH -J build_paragram
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:30:00
set -euo pipefail
P=/raven/u/uli/libbench-src/paragram
COMMIT=4f36b1f2c8738912c5efc056a06023d4c1d2da07  # main 2026-09-01 (package 0.1.0)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "${SLURM_SUBMIT_DIR:-}" ] && here="$SLURM_SUBMIT_DIR/bench/libraries"

if [ "${1:-}" = fetch ]; then
    mkdir -p "$P"
    cd "$P"
    [ -d paragram-src ] || git clone https://github.com/zenseact/paragram.git paragram-src
    git -C paragram-src checkout "$COMMIT"
    git -C paragram-src submodule update --init  # cuBQL d18c5fa
    [ -d venv ] || /mpcdf/soft/SLE_15/packages/x86_64/python-waterboa/2025.06/bin/python3 -m venv venv  # Python 3.13.5
    venv/bin/pip install --upgrade pip
    venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    venv/bin/pip install numpy ninja packaging rich
    exit 0
fi

source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$P/venv/bin:$PATH" TORCH_EXTENSIONS_DIR="$P/torch_ext" TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=${SLURM_CPUS_PER_TASK:-16} VERBOSE=1
export PYTHONPATH="$P/paragram-src"
"$P/venv/bin/python" -c "
import torch, paragram, paragram.cuda
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name())
import paragram as pg
p = torch.rand(100000, 3, device='cuda')
d = pg.voronoi_diagram(p)
print('smoke: adjacency', d.adjacency.shape, 'status nonzero', int((d.status != 0).sum()))
from paragram import _backend
import importlib, sys
sys.argv = ['x']
spec = importlib.util.spec_from_file_location('drv', '$here/paragram.py'); drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
drv.ours(); print('volume kernel built')
"
