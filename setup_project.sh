#! /bin/bash
set -e

########### ENVIRONMENT VARIABLES ##############
### Set before running the rest of the setup ###
BIOMARKERS_PROJECT_HOME=/mnt/c/Users/tobit/Documents/Doctorado/Initiative1_epic1/Project
CHB_DATASET_HOME=/mnt/c/Users/tobit/Downloads/eeg_datasets/chb-mit
SIENA_DATASET_HOME=/mnt/c/Users/tobit/Downloads/eeg_datasets/siena
TUSZ_DATASET_HOME=/mnt/c/Users/tobit/Downloads/eeg_datasets/tusz

grep -qF -- "BIOMARKERS_PROJECT_HOME" ~/.bashrc || echo -e "\nexport BIOMARKERS_PROJECT_HOME=${BIOMARKERS_PROJECT_HOME}" >> ~/.bashrc
grep -qF -- "CHB_DATASET_HOME" ~/.bashrc || echo -e "export CHB_DATASET_HOME=${CHB_DATASET_HOME}" >> ~/.bashrc
grep -qF -- "TUSZ_DATASET_HOME" ~/.bashrc || echo -e "export SIENA_DATASET_HOME=${SIENA_DATASET_HOME}" >> ~/.bashrc
grep -qF -- "TUSZ_DATASET_HOME" ~/.bashrc || echo -e "export TUSZ_DATASET_HOME=${TUSZ_DATASET_HOME}" >> ~/.bashrc
source ~/.bashrc

# Install ubuntu packages for building python
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv
grep -qF -- "PYENV_ROOT" ~/.bashrc || curl https://pyenv.run | bash
grep -qF -- "PYENV_ROOT" ~/.bashrc || echo -e '\nexport PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
grep -qF -- "pyenv init" ~/.bashrc || echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
pyenv --version

# Install python version
pyenv install 3.11.0

# Select python version
pyenv init
pyenv shell 3.11.0

# Create virtual environment
python -m venv /var/tmp/venv-project-1
source /var/tmp/venv-project-1/bin/activate
pip install --upgrade pip
python -m pip install -t requirements.txt
deactivate

# Create directory tree