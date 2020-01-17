#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 60G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -n 10
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o cellfinder.out
#SBATCH -e cellfinder.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail@domain.com

cell_file='/path/to/signal/channel'
background_file='path/to/background/channel'
output_dir='/path/to/output/directory'

echo "Loading conda environment"
module load miniconda
conda activate cellfinder

echo "Running cellfinder"
cellfinder -s $cell_file -b $background_file -o $output_dir --metadata metadata_file.yml --summarise
