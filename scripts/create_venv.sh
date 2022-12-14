#!/bin/sh

# "Template Repository for Research Papers with Python Code"
#
# Copyright (C) 2022 Peter Steiner
# License: BSD 3-Clause

python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
deactivate
