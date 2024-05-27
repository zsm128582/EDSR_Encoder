#!/bin/bash
#SBATCH --time=1440
#SBATCH -p g078t1
#SBATCH --gres=gpu:4
####这个是扣费的项目
#SBATCH --comment=vno2024
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --output=/home/u2023110769/code/EDSR_Encoder/%j.out
#SBATCH --chdir=/home/u2023110769/code/EDSR_Encoder
#SBATCH --job-name=randomN


source ~/.bashrc
# 确保singularity可用
module load singularity

echo job start time is `date`
/opt/app/singularity/bin/singularity   exec -B /opt/app/nvidia/535.154.05/lib/  --nv /home/u2023110769/for_edsr.sif /bin/bash -l -c "  /opt/miniconda3/bin/conda init bash ; source ~/.bashrc ; export LD_LIBRARY_PATH=/opt/app/nvidia/535.154.05/lib/:$LD_LIBRARY_PATH ;conda activate myEncoder ; pip list  ;  nvidia-smi ; cd /home/u2023110769/Code/EDSR_Encoder ;  python /home/u2023110769/Code/EDSR_Encoder/train.py --config configs/train-randomN.yaml"
echo job end time is `date`