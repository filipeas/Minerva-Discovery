#!/bin/bash

# Defina o nome do ambiente Conda
ENV_NAME="sam_v1"

# Passo 1: Verificar se o ambiente Conda já existe
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Ambiente Conda '$ENV_NAME' não encontrado. Criando agora..."
    # Passo 2: Criar o ambiente Conda (caso não exista)
    module load anaconda3
    conda create -n "$ENV_NAME" python=3.10 -y
else
    echo "Ambiente Conda '$ENV_NAME' já existe."
fi

# Passo 3: Ativar o ambiente Conda
echo "Ativando o ambiente Conda '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Passo 4: Instalar os pacotes necessários
echo "Instalando pacotes..."
pip install -r ../../requirements.txt
cd ../../Minerva-Dev && pip install .

# Configurações do SLURM para submissão do job
#SBATCH --job-name=sam_FAS
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu -i
#SBATCH --time=120:00:00
#SBATCH -o out_out.txt
#SBATCH -e out_err.txt

# Comandos do job SLURM
cd $SLURM_SUBMIT_DIR
original_dir=$(pwd)

module load hwloc
module load cuda/12.1-gcc-12.2.0-pig3lo5
module load anaconda3
unset LD_LIBRARY_PATH

echo "Moving files to scratch"
mkdir /scratch/local/$ENV_NAME
cp experiment_sam.py /scratch/local/$ENV_NAME
cp config_experiment_sam.json /scratch/local/$ENV_NAME
cd /scratch/local/$ENV_NAME

echo "Starting job"
srun -u conda run --no-capture-output -n $ENV_NAME python -u experiment_sam.py --config config_experiment_sam.json

echo "Moving files back"
cp -rf * $original_dir
cd ../
rm -rf /scratch/local/sam_v1