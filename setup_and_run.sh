#!/bin/bash

# Verificando se a flag --container foi fornecida
if [[ "$1" == "--container" ]]; then
    echo "installing Minerva-Dev..."
    cd Minerva-Dev
    pip install .

    echo "installing requirements..."
    cd ..
    pip install -r requirements.txt
    
    echo "Container is up!"
else
    echo "creating virtual environment..."
    rm -rf experiment
    python -m venv experiment

    echo "Activating virtual environment..."
    source experiment/bin/activate

    echo "Updating pip..."
    pip install --upgrade pip

    echo "installing Minerva-Dev..."
    cd Minerva-Dev
    pip install .

    echo "installing requirements..."
    cd ..
    pip install -r requirements.txt

    echo "executing main.py..."
    cd my_experiments
    python main.py
fi