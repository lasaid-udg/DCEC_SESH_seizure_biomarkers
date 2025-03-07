#! /bin/bash
set -e

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
BIOMARKERS_PROJECT_HOME="$(head -1 .env | cut -d "=" -f 2)"
mkdir ${BIOMARKERS_PROJECT_HOME}/data
mkdir ${BIOMARKERS_PROJECT_HOME}/docs
mkdir ${BIOMARKERS_PROJECT_HOME}/images
mkdir ${BIOMARKERS_PROJECT_HOME}/images/chb-mit
mkdir ${BIOMARKERS_PROJECT_HOME}/images/siena
mkdir ${BIOMARKERS_PROJECT_HOME}/images/tusz
mkdir ${BIOMARKERS_PROJECT_HOME}/slices/
mkdir ${BIOMARKERS_PROJECT_HOME}/slices/chb-mit
mkdir ${BIOMARKERS_PROJECT_HOME}/slices/siena
mkdir ${BIOMARKERS_PROJECT_HOME}/slices/tusz
mkdir ${BIOMARKERS_PROJECT_HOME}/windows/
mkdir ${BIOMARKERS_PROJECT_HOME}/windows/chb-mit
mkdir ${BIOMARKERS_PROJECT_HOME}/windows/siena
mkdir ${BIOMARKERS_PROJECT_HOME}/windows/tusz
mkdir ${BIOMARKERS_PROJECT_HOME}/reports/
mkdir ${BIOMARKERS_PROJECT_HOME}/features/
mkdir ${BIOMARKERS_PROJECT_HOME}/features/chb-mit
mkdir ${BIOMARKERS_PROJECT_HOME}/features/siena
mkdir ${BIOMARKERS_PROJECT_HOME}/features/tusz