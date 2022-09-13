#!/bin/sh

# "Feature Engineering and Stacked Echo State Networks for Musical Onset Detection"
#
#  Copyright (C) 2022 Peter Steiner
# License: BSD 3-Clause

python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter-lab

deactivate
