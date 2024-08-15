#Imagen que nos da NVIDIA
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y wget git sox libsox-fmt-all

# Descargar e instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    # Crea un directorio .conda en el directorio home del usuario root.
    && mkdir /root/.conda \
    # Ejecuta el script de instalación de Miniconda en modo silencioso (sin interacción del usuario, con -b).
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    # Elimina el script de instalación después de que Miniconda ha sido instalado para mantener limpio el entorno Docker.
    && rm -f /tmp/miniconda.sh

#cd /root
WORKDIR /root

# Copiar el archivo environment.yml al directorio de trabajo
COPY environment.yml .

# Actualizar conda e instalar pip
RUN conda update -n base -c defaults conda -y
RUN conda install pip -y

# Crear el entorno conda 'ggvad' a partir del archivo environment.yml
RUN conda env create -f environment.yml

#DEPENDENCIAS O INSTALACION PARA EL PROYECTO "TARATARATARA" *********************
SHELL ["conda", "run", "-n", "diffuse", "/bin/bash", "-c"]

# Estas instalaciones con PIP se hicieron veloz
RUN pip install pydub praat-parselmouth essentia TextGrid
RUN conda install -c conda-forge ffmpeg

#Esta instalacion la estoy haciendo para python 3.9:
RUN conda install -c anaconda h5py
RUN pip install bvhsdk
RUN pip install wandb
RUN pip install notebook