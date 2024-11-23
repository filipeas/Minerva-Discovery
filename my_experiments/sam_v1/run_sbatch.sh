#!/bin/bash
#SBATCH --partition=gpulongd       # Nome da partição que você vai usar (ex: gpulongd)
#SBATCH --account=asml-gpu         # Conta associada
#SBATCH --job-name=sam_FAS  # Nome do job
#SBATCH --time=120:00:00           # Tempo máximo para o job (120 horas ou 5 dias)

# Exibe informações do job para facilitar o debug
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# # vai pra raiz
# cd ../..

# echo "installing requirements..."
# pip install -r requirements.txt

# echo "installing Minerva-Dev..."
# cd Minerva-Dev
# pip install .

# # volta pra pasta original
# cd -

# Executa o container Singularity com suporte a GPU e roda o script de setup
singularity exec --nv ../../Singularity.sif bash execute.sh