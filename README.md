# MAPPO_AVs
MAPPO-based traffic control

## Install

1. Prepare a python environment, e.g.:
```
$ conda create -n metadrive-mappo python=3.8 -y
$ conda activate metadrive-mappo
```

2. Clone the repo:
```
$ git clone git@github.com:alstondu/MAPPO-AVs.git

```

3. Install required packages for mappo:
```
$ cd PATH/TO/MAPPO-AVs/mappo
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install -r requirements.txt
```

4. Install MetaDrive:
```
$ cd PATH/TO/MAPPO-AVs/metadrive
$ pip install -e .
```

4. Start training :
```
$ python train/train.py
```
