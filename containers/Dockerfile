FROM python:3.11-slim

WORKDIR /workspace

COPY calvin_utils_requirements.txt .

RUN pip install --no-cache-dir -r calvin_utils_requirements.txt
RUN pip install jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
