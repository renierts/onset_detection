#!/bin/sh

# "Template Repository for Research Papers with Python Code"
#
# Copyright (C) 2022 Peter Steiner
# License: BSD 3-Clause

python3 -m venv .virtualenv
source .virtualenv/bin/activate
python3 -m pip install -r requirements.txt
jupyter-lab

deactivate
