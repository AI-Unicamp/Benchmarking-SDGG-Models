#Image that provides us with NVIDIA CUDA on Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y wget git sox libsox-fmt-all

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    # Create a .conda directory in the root user's home directory.
    && mkdir /root/.conda \
    # Run the Miniconda installation script in silent mode (without user interaction, using -b).
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    # Delete the installation script after Miniconda has been installed to keep the Docker environment clean.
    && rm -f /tmp/miniconda.sh

#cd /root
WORKDIR /root

# Copy the environment.yml file to the working directory.
COPY environment.yml .

# Update conda and install pip.
RUN conda update -n base -c defaults conda -y
RUN conda install pip -y

# Create the conda environment 'ggvad' from the environment.yml file.
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "sdgg", "/bin/bash", "-c"]

# These installations with PIP were done quickly.
RUN pip install pydub praat-parselmouth essentia TextGrid
RUN conda install -c conda-forge ffmpeg

# I am performing this installation for Python 3.9.
RUN conda install -c anaconda h5py
RUN pip install bvhsdk
RUN pip install wandb
RUN pip install notebook
