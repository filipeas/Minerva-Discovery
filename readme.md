# Minerva-Discovery
- Experiments using Minerva-Dev!

## How clone this repositore
1) First, head this README.md!
2) In your machine, execute:
    - ``` git clone https://github.com/filipeas/Minerva-Discovery.git ```
    - ``` cd Minerva-Discovery ```
    - ``` git submodule update --init --recursive ```

## How to execute in local (for test)
- run ``` ./setup_and_run.sh ```

## How execute into Ogbon (Petrobras)
1) Read this README.md file first!
2) Make sure you have Singularity on server.
3) Open **Singularity.def** and see your configuration. Add new packeges into environment, if you need.
4) Add new packages in **requirements.txt**, if you need.
5) Build a image: 
    - ``` singularity build Singularity.sif Singularity.def ```. 
    Make sure rename labels **sam_FAS** to a name that you want (check into **Singularity.def**, because there is a flag called **Maintainer** with the same name).
6) (optional) You can run singularity image with: 
    - ``` singularity exec --nv Singularity.sif bash ```. 
    - But, in Ogbon (Petrobras), you need run with Slurm, SO, BE CAREFUL!
7) (optional) In Ogbon (Petrobras), execute:
    - ``` srun --partition gpulongd --account asml-gpu --job-name=sam_FAS --time 120:00:00 --pty bash ```. 
    - This will open a bash for use Slurm. So, after this, run **step 6)**.
8) (opcional) If you need run using sbatch, use this: 
    - ``` sbatch --output=meu_job.log --error=meu_job.err run_sbatch.sh ```.

## Tips
- In Ogbon (Petrobras), execute ``` srun --partition gpulongd --account asml-gpu --job-name=NOME_DO_PROJETO_OU_EXPERIMENTO --time 120:00:00 --pty bash ``` for interative bash.
- More details, see:
    - https://hpc.senaicimatec.com.br/docs/clusters/job-scheduling/

## How to run with TMUX
- Access some node (discovery) and execute: ``` tmux new -s sam_FAS ```
- Go to directory (your-directory) and execute: ``` python main.py ```
- After this, exit of session executing this: ``` Ctrl + B ``` and press ``` D ```.
- To access the session, execute: ``` tmux attach -t sam_FAS ```
- To list TMUX sessions, execute: ``` tmux ls ```