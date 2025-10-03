#!/bin/bash
#SBATCH --partition=a800x8
#SBATCH -J test
#SBATCH --ntasks-per-node=1       
#SBATCH --gres=gpu:1             
#SBATCH -o job.%j.out                  
#SBATCH -e job.%j.err    

source activate OpenSTL
python test.py
