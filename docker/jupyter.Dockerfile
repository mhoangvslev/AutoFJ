FROM jupyter/scipy-notebook
#RUN apt-get update && apt-get install -y wget curl git build-essential

RUN pip install ipykernel jupyterlab jupyter_http_over_ws \
    && jupyter serverextension enable --py jupyter_http_over_ws

WORKDIR /content/
COPY . /content/