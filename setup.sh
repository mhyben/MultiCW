#!/bin/bash

# Ask user for the Conda installation path
read -p "Enter the full path to your Conda installation (e.g., /home/yourname/miniconda3): " CONDA_ROOT

# Construct path to conda.sh
CONDA_SETUP="$CONDA_ROOT/etc/profile.d/conda.sh"

# Check if the file exists
if [ -f "$CONDA_SETUP" ]; then
    source "$CONDA_SETUP"
else
    echo "❌ Conda setup script not found at: $CONDA_SETUP"
    echo "Please ensure you provided the correct path to your Conda installation."
    exit 1
fi

# Function to set up an environment
create_env() {
    ENV_NAME=$1
    REQUIREMENTS_FILE=$2

    echo "🔧 Creating environment: $ENV_NAME"
    conda create --yes --name "$ENV_NAME" python=3.10
    conda activate "$ENV_NAME"
    conda install --yes -c anaconda ipykernel
    python -m ipykernel install --user --name="$ENV_NAME"
    conda install --yes -n base -c conda-forge jupyterlab_widgets

    if [ "$REQUIREMENTS_FILE" != "none" ]; then
        pip install -r "$REQUIREMENTS_FILE"
    fi

    python -m spacy download en_core_web_sm
    conda deactivate
}

# Setup MultiCW-dataset environment
echo "📦 Setting up environment: MultiCW-dataset"
conda create --yes --name MultiCW-dataset python=3.10
conda activate MultiCW-dataset
pip install jupyterlab
conda install --yes -c anaconda ipykernel
python -m ipykernel install --user --name=MultiCW-dataset
conda install --yes -c conda-forge gliner
conda install --yes -n base -c conda-forge jupyterlab_widgets
jupyter labextension install js
conda deactivate

# Setup MultiCW-finetune environment
create_env "MultiCW-finetune" "requirements-finetune.txt"

# Setup MultiCW-lesa environment
create_env "MultiCW-lesa" "requirements-lesa.txt"
