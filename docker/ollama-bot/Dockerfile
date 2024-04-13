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
RUN /env/bin/pip install ollama
RUN /env/bin/pip install discord.py

ENTRYPOINT /env/bin/python runner.py
