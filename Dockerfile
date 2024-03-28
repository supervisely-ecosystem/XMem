FROM supervisely/xmem:1.0.2

COPY . /repo

RUN pip install -r /repo/requirements.txt

RUN pip install git+https://github.com/supervisely/supervisely.git@bbox_tracking_debug
