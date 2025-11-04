# The required base image
FROM ubuntu:22.04

# Set non-interactive mode for apt-get - avoids prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3, pip, venv, and git
# Clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /gym-gridbot

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Install the application in editable mode
# This allows for changes in the source code to be reflected without reinstalling
RUN pip3 install -e .

# Define the command to run the application
CMD ["python3", "--version"]
