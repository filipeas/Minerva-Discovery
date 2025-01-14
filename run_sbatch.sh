#!/bin/bash
#SBATCH --partition=gpulongd       # Nome da partição que você vai usar
#SBATCH --account=asml-gpu         # Conta associada
#SBATCH --job-name=sam_FAS         # Nome do job
#SBATCH --time=220:00:00           # Tempo máximo para o job (em horas)
#SBATCH --mem=200G                 # Quantidade mínima de RAM
#SBATCH --constraint=gpumem48      # Garantir GPU com no mínimo 48 GB de VRAM

# Exibe informações do job para facilitar o debug
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Executa o container Singularity com suporte a GPU e roda o script de setup
singularity exec --nv Singularity.sif \ 
    python my_experiments/sam_original/exec_experiment_2/_with_prompt/main.py --config config_experiment_parihaka_ogbon.json