#!/bin/sh

module load python2

virtualenv ~/minstgan_env

source minstgan_env /bin/activate

pip install torch
pip install torchvision
pip install numpy
pip install matplotlib
pip install jupyter
