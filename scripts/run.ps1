# "Feature Engineering and Stacked Echo State Networks for Musical Onset Detection"
#
#  Copyright (C) 2022 Peter Steiner
# License: BSD 3-Clause

python.exe -m venv venv

.\venv\Scripts\activate.ps1
python.exe -m pip install -r requirements.txt
python.exe .\src\main.py --frame_sizes 1024 2048 4096 --num_bands 3 6 12

deactivate
