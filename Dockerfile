# Creating a Dockerfile to run the Isaac-GROOT by NVIDIA
# Importing nvidia/cuda image with CUDA12.4 support 
# Making sure we have Ubuntu image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Setting the frontend to be non-interactive
# This is to avoid any user input required during the installation of packages
ENV DEBIAN_FRONTEND=noninteractive

# Updating repositories and installing necessary packages for installing Conda and Python Libraries
RUN apt-get update && apt-get install wget -y && apt-get install -y python3-pip\
        ffmpeg \
        libopencv-dev \
        git

RUN ln -s /usr/bin/python3 /usr/bin/python

# Clone the GR00T repo from the NVIDIA GitHub Page
RUN git clone https://github.com/nvidia/Isaac-GR00T.git /GR00T

# Entering the GR00T directory
WORKDIR /GR00T

# Now run all the conda related commands to make sure conda gets initialized in the bash script
# Install all necessary python packages for gr00t
RUN <<EOT 
pip install --upgrade setuptools
pip install .
pip install --no-build-isolation flash-attn==2.7.1.post4
pip install google-cloud-storage jsonlines
pip install notebook
EOT

# Setting the Entrypoint and Command to /bin/bash whenever executed
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["sleep infinity"]
# ENTRYPOINT ["/bin/bash"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]