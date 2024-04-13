FROM ubuntu:jammy

WORKDIR /code

# Get latest apt updates and upgrades
RUN apt update
RUN apt upgrade -y

# Install python3
RUN apt install -y python3-full

# Create new virtual environment to install dependencies into
RUN python3 -m venv /env

# Install pip packages and deps
RUN /env/bin/pip install --upgrade pip
RUN /env/bin/pip install diffusers
RUN /env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN /env/bin/pip install transformers
RUN /env/bin/pip install accelerate
RUN /env/bin/pip install safetensors
RUN /env/bin/pip install discord.py
RUN /env/bin/pip install matplotlib

ENTRYPOINT /env/bin/python runner.py 
