FROM python:3.8

WORKDIR /crawl

RUN apt-get update && apt-get install -y libtinfo5

RUN git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client

RUN wget https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/master/crawlingathome.py
RUN wget https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/master/requirements.txt
RUN wget https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/master/clip_filter.py

RUN wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/blocklists/blocklist-domain.txt
RUN wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/blocklists/failed-domains.txt
RUN wget https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/blocklists/5Mduplicates.txt

RUN pip3 install wheel --no-cache-dir

RUN pip3 install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

RUN pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
RUN pip3 install -r ./requirements.txt --no-cache-dir

RUN pip3 install tensorflow --no-cache-dir
RUN pip3 install git+https://github.com/openai/CLIP --no-cache-dir

RUN pip3 install -U --force-reinstall pdbpp --no-cache-dir
RUN pip3 install --force-reinstall msgpack==1.0.1 --no-cache-dir

RUN yes | pip3 uninstall pillow
RUN CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd --no-cache-dir

RUN yes | pip3 uninstall asks
RUN pip3 install git+https://github.com/rvencu/asks --no-cache-dir

RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip
RUN unzip -o protoc-3.17.3-linux-x86_64.zip -d /usr bin/protoc
RUN unzip -o protoc-3.17.3-linux-x86_64.zip -d /usr 'include/*'
RUN rm -f protoc-3.17.3-linux-x86_64.zip

ENV PYTHONHASHSEED=0
ENV NAME="ARKseal"

#CMD touch ./crawl.log && python3 -u crawlingathome.py --name $NAME >> ./crawl.log