#!/bin/bash
. /etc/profile
source /home/fcaporaso/.bashrc

conda activate env_py

python fit_pairs_profile.py -ncores 20 -file profile_1.fit &
python fit_pairs_profile.py -ncores 20 -file profile_1.fit &
python fit_pairs_profile.py -ncores 20 -file profile_1.fit &
wait
