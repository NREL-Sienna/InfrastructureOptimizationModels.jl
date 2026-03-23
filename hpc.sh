#!/bin/bash
#SBATCH --account=siennadev
#SBATCH --time=5:00:00
#SBATCH --job-name=ac_network_test
#SBATCH --mail-user=anthony.costarelli@nlr.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=hpc_output/%j.out

module load julia
julia --project=test scripts/bilinear_delta_benchmark_hpc.jl