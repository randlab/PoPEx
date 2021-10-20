#!/bin/bash
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --time=4-00:00:00
#SBATCH --partition=bigmem
#SBATCH --mem=0

python run-popex.py 1> run.out 2> run.err
