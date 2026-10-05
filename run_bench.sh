#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="swiftpack-bench"
#SBATCH --nodes=1
#SBATCH --output="swiftpack-bench.%j.%N.out"
#SBATCH -t 00:45:00
##################### SLURM (do not change) ^  #####################

# create python virtual environment
python3 -m venv $(pwd)/.venv
source $(pwd)/.venv/bin/activate
pip install -r $(pwd)/requirements.txt
# mkdir -p $(pwd)/plots

python3 $(pwd)/benchmarks/matmul_numba_bench.py 512 $(pwd)/doc/matmul_numba_bench_512.png
