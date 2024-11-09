#!/bin/bash

# vai pra raiz
cd ../..

echo "installing requirements..."
pip install -r requirements.txt

echo "installing Minerva-Dev..."
cd Minerva-Dev
pip install .

# volta pra pasta original
cd -
python experiment_sam.py --config config_experiment_sam.json