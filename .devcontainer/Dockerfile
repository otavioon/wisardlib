FROM ubuntu:22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install common apt packages
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update &&  \
    apt install -y  build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        curl \
        git \
        htop \
        libssl-dev \
        nvtop \
        net-tools \
        python3-all  \
        python-is-python3 \
        python3-pip \
        python3-venv \
        tmux \
        tree \
        unrar \
        unzip \
        vim \
        wget \
        xdot

# Create the vscode user (non-root user) and add sudo support
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME  && \
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Change group to same as system
RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
RUN     rm -rf /var/lib/apt/lists/* && \
        ldconfig && \
        apt autoremove && \
        apt clean

# ***************** Specific packages ********************

# System-wide python packages
RUN python -m pip install --upgrade pip setuptools pytest && \
    python -m pip install black \
        ipykernel \
        kaleido \
        matplotlib \
        nbformat \
        numpy \
        pandas \
        plotly \
        seaborn \
        scipy 

# Change shell for user b
RUN chsh --shell /bin/bash $USERNAME
#  Set the default user
USER $USERNAME

