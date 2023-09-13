#! /bin/bash
#SBATCH --verbose
#SBATCH --partition gpu
#SBATCH --job-name fever-t
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=yx2433@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu7

#SBATCH --array=1
#SBATCH --output=sbl_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=sbl_%A_%a.err

echo === $(date)
#module load anaconda3



export CUDA_DEVICE=0
#source activate /gpfsnyu/home/yx2433/anaconda3/envs/fever
conda info --envs
nvidia-smi
pwd
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo ${SLURM_ARRAY_TASK_ID}

set -ex

python scripts/fact_verification.py \
  --model_name_or_path /gpfsnyu/home/yx2433/transformers/models/albert-base-v2 \
  --tasks_names vitaminc \
  --overwrite_output_dir \
  --do_train \
  --do_test \
  --eval_all_checkpoints \
  --data_dir data \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --learning_rate 2e-5 \
  --max_steps 50000 \
  --save_step 10000 \
  --overwrite_cache \
  --output_dir results/vitaminc_albert_base \
  "$@"

  #--fp16 \
  #--test_tasks vitc_real vitc_synthetic \
  #--do_train \
  #--do_predict \
  #--test_on_best_ckpt \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
