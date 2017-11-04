#!/bin/bash

cd ~

# Download and install anaconda3.
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
chmod 774 Anaconda3-5.0.1-Linux-x86_64.sh 
./Anaconda3-5.0.1-Linux-x86_64.sh

# Answer prompts (yes, press 'q', yes, etc...)

# Download and install cuda repo.
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

# Clean up.
rm Anaconda3-5.0.1-Linux-x86_64.sh cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

cd -

# Install pip3, cuda, pymongo (for BSON), PIL, and pytorch.
sudo apt-get update --fix-missing
sudo apt-get install cuda-8-0
sudo apt-get install python3-pip
sudo pip3 install pymongo
sudo pip3 install Pillow
sudo conda install pytorch torchvision cuda80 -c soumith
