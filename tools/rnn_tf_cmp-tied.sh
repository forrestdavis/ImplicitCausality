#!/bin/bash
#SBATCH --job-name=event
#SBATCH --output=/home/fd252/ImplicitCausality/models/logs/log-rnn_context-tied.%a.out
#SBATCH --error=/home/fd252/ImplicitCausality/models/logs/log-rnn_context-tied.%a.err
#SBATCH --ntasks=1
#SBATCH --mem=15000
#SBATCH --time=2-00:00:00
#SBATCH --partition=default_gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=fd252@cornell.edu
#SBATCH --array=102-110

activate env
#source activate

Lflag=$SLURM_ARRAY_TASK_ID

model='LSTM';
dropout=0.2;
corpussize='80m';
nhid='400';

seed=$(($Lflag % 100));

workdir='/home/fd252/ImplicitCausality'

# Train
time python -u "/home/fd252/neural-complexity/main.py" --model ${model} --model_file "${workdir}/models/wikitext-103_${model}_${nhid}_${seed}-d${dropout}.pt" --vocab_file "wikitext103_vocab" --data_dir "${workdir}/wikitext-103/" --trainfname "wiki.train.tokens" --validfname "wiki.valid.tokens" --nhid ${nhid} --epochs 40 --emsize ${nhid} --seed ${seed} --dropout ${dropout} --lowercase --cuda --tied > "${workdir}/models/logs/${model}_${nhid}_${seed}-d${dropout}.training"
