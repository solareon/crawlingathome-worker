#!/bin/bash

sudo apt update && sudo apt install -y git build-essential python3.7-dev python3-pip python3.7-venv libjpeg-dev
python3 -m venv venv && . venv/bin/activate

rm requirements.txt blocklist-domain.txt crawlingathome.py clip_filter.py
rm -r crawlingathome_client

git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
wget https://raw.githubusercontent.com/rvencu/crawlingathome-worker/colab-mod-asks/requirements.txt
wget https://raw.githubusercontent.com/Wikidepia/crawlingathome-worker/master/blocklist-domain.txt
wget https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/split-tractor-trio/crawlingathome.py
wget https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/split-tractor-trio/clip_filter.py

pip3 install torch==1.7.1$1 torchvision==0.8.2$1 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

#pip3 install git+https://github.com/rvencu/asks # breaks in colab
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r ./requirements.txt --no-cache-dir

pip3 install datasets ftfy pandas pycld2 regex tfr_image tractor trio ujson --no-cache-dir
pip3 install tensorflow --no-cache-dir
pip3 install git+https://github.com/openai/CLIP --no-cache-dir

pip3 install -U --force-reinstall pdbpp --no-cache-dir
pip3 install --force-reinstall msgpack==1.0.1 --no-cache-dir

yes | pip3 uninstall pillow
CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd --no-cache-dir