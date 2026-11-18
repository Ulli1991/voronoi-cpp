#!/bin/bash
# A Python script of this repository with numpy, off the login node:
#   srun -p interactive -n1 -c1 --mem=8G --time=00:10:00 scripts/step2x_python.sh <script> [args]
source /etc/profile.d/modules.sh
module purge
module load python-waterboa/2025.06
python3 "$@"
