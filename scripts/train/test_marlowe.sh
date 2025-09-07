#!/bin/sh 
#SBATCH --job-name=test_marlowe
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000123
#SBATCH -G 1
#SBATCH --time=00:01:00
#SBATCH --output=test_marlowe.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu

module load slurm 
module load nvhpc 
module load cudnn/cuda12/9.3.0.75 

echo "SUCCESS"