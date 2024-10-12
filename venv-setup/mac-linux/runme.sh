#!/bin/bash

echo "starting the installation process"

#build a python3 venv called ais
python3 -m venv ais

#activates the ais venv
source ais/bin/activate

#installs requirements.txt into your venv
pip install -r requirements.txt

echo "Setup complete! the venv ais has the dependencies in requirements.txt installed"

#deactivates ais
deactivate
