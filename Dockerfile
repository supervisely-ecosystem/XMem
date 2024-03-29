FROM supervisely/xmem:1.0.2

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY supervisely_integration/serve/requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY . /app
RUN pip install git+https://github.com/supervisely/supervisely.git@bbox_tracking_debug

EXPOSE 80

ENV PYTHONPATH "${PYTHONPATH}:/app/supervisely_integration/serve/src"
ENV APP_MODE=production ENV=production

ENTRYPOINT ["python3", "-u", "-m", "uvicorn", "supervisely_integration.serve.src.main:model.app"]
CMD ["--host", "0.0.0.0", "--port", "80"]
