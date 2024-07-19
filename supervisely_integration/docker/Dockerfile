FROM supervisely/base-py-sdk:6.72.87
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN python3 -m pip install packaging==24.1

RUN pip3 install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

RUN python3 -m pip install supervisely==6.73.102

RUN mkdir -p weights
COPY XMem.pth /weights/
COPY resnet18.pth /weights/
COPY resnet50.pth /weights/
