#!/bin/bash
set -e 


# this code works 
# conda create -y -n retriever python=3.10
# source /opt/conda/etc/profile.d/conda.sh
# conda activate retriever
# which python

# # we recommend installing torch with conda for faiss-gpu
# conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install transformers datasets pyserini

# ## install the gpu version faiss to guarantee efficient RL rollout
# conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0

# # ## API function
# pip install uvicorn fastapi





# #!/bin/bash
# set -e 
conda create -y -n retriever python=3.10
source /opt/conda/etc/profile.d/conda.sh
conda activate retriever
which python

pip install uv

# we recommend installing torch with conda for faiss-gpu
mamba install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
uv pip install transformers datasets pyserini

# ## install the gpu version faiss to guarantee efficient RL rollout
mamba install -y -c pytorch -c nvidia faiss-gpu=1.8.0

# ## API function
uv pip install uvicorn fastapi






