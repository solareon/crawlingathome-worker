FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
ARG PYTHON_VERSION=3.8

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
          python${PYTHON_VERSION} \
          python3-pip \
          python${PYTHON_VERSION}-dev \
          wget \
# Change default python
    && cd /usr/bin \
    && ln -sf python${PYTHON_VERSION}         python3 \
    && ln -sf python${PYTHON_VERSION}m        python3m \
    && ln -sf python${PYTHON_VERSION}-config  python3-config \
    && ln -sf python${PYTHON_VERSION}m-config python3m-config \
    && ln -sf python3                         /usr/bin/python \
# Update pip and add common packages
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade \
        setuptools \
        wheel \
        six \
# Cleanup
    && apt-get clean \
    && rm -rf $HOME/.cache/pip

WORKDIR /crawl

RUN apt-get update && apt-get upgrade -y && apt-get install -y git build-essential libtinfo5 libjpeg-dev python3-dev python3-pip python3-venv rsync

RUN git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client && \
wget https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/master/crawlingathome.py && \
wget https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/master/requirements.txt

RUN python3 -m venv venv && . venv/bin/activate && \
python3 -m pip install -U pip && \
pip3 install wheel --no-cache-dir && \
\
pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir && \
\
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir && \
pip3 install -r ./requirements.txt --no-cache-dir && \
\
pip3 install tensorflow --no-cache-dir && \
\
pip3 install clip-anytorch --no-cache-dir && \
\
yes | pip3 uninstall pillow && \
CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd --no-cache-dir && \
\
yes | pip3 uninstall asks && \
pip3 install git+https://github.com/rvencu/asks --no-cache-dir && \
\
yes | pip3 uninstall protobuf && \
pip3 install protobuf==3.9.2

RUN wget https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/master/clip_filter.py && \
wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/blocklists/blocklist-domain.txt && \
wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/blocklists/failed-domains.txt && \
wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/blocklists/5Mduplicates.txt

ENV PYTHONHASHSEED=0
ENV NAME="ARKseal"

CMD . venv/bin/activate && nice python3 -u crawlingathome.py --name $NAME >> /dev/stdout
