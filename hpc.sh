#!/bin/bash
#SBATCH --account=siennadev
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --job-name=ac_network_test
#SBATCH --mail-user=anthony.costarelli@nlr.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=hpc_output/%j.out
#SBATCH --error=hpc_output/%j.err

set -e

echo "=== Job diagnostics ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================"

module load julia

echo "Julia:     $(which julia)"
echo "Julia ver: $(julia --version)"
echo "========================"

# stdbuf -oL forces line-buffered stdout so output is flushed incrementally
# instead of being lost if the process is killed by SLURM time limit.
stdbuf -oL julia --project=test scripts/bilinear_delta_benchmark_hpc.jl

echo "Finished:  $(date)"