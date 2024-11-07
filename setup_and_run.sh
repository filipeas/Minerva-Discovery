#!/bin/bash
echo "creating virtual environment..."
rm -rf experiment
python -m venv experiment

echo "Activating virtual environment..."
source experiment/bin/activate

echo "Updating pip..."
pip install --upgrade pip

echo "installing requirements..."
pip install -r requirements.txt

echo "installing Minerva-Dev..."
cd Minerva-Dev
pip install .

echo "executing main.py..."
cd ../my_experiments
python main.py