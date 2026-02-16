FROM --platform=linux/amd64 python:3.10

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyterlab

ENV JUPYTER_PASSWORD=whowlab

CMD bash -c "jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.password=\"\$(python -c 'from jupyter_server.auth import passwd; print(passwd(\"whowlab\"))')\" \
  --ServerApp.token=''"