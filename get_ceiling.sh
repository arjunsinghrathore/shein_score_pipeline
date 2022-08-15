#!/bin/bash
#SBATCH --time= 2:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --account=carney-tserre-condo
#SBATCH -J get_ceiling_TRAIN_TEST
# Specify an output file
#SBATCH -o get_ceiling_TRAIN_TEST.out
#SBATCH -e get_ceiling_TRAIN_TEST.out



#echo ". /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
export PATH="/gpfs/runtime/opt/anaconda/3-5.2.0/bin:$PATH"
export RESULTCACHING_DISABLE=1
conda deactivate
conda activate arjun_bs
cd /media/data_cifs/projects/prj_brainscore/arjun_brainscore/bs_hackathon

python -u get_ceiling.py --train_size=0.5 --test_size=0.5 --assembly_name=sheinberg.neural.IT #.4more
