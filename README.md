# MAPPO for Autonomous Vehicles (MAPPO-AVs)

This project utilizes the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm, a multi-agent reinforcement learning (MARL) approach, to train autonomous driving agents in traffic scenarios using the MetaDrive environment. The goal is to enable multiple autonomous driving agents to learn collaboration and policy optimization in complex traffic environments, improving traffic safety, success rate, and efficiency.

## Installation

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
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ cd PATH/TO/MAPPO-AVs/mappo
$ pip install -r requirements.txt
```

4. Replace the modified file in the MetaDrive lib:
```
$ cp torch_expert.py $CONDA_PREFIX/lib/python3.8/site-packages/metadrive/examples/ppo_expert/torch_expert.py
```

## Usage

### Training

The defalt training configuration trains the MAPPO64 in the roundabount map, which contains 20 vehicls for 1M timesteps:
```
$ cd PATH/TO/MAPPO-AVs/mappo
$ python train/train.py
```
Models and logs will be saved in the corresponding path under the `results/` directory.

### Evaluation
To perform evaluation using a trained model, use the `--eval` flag:
```
$ python train/train.py --eval --eval_model_dir /path/to/model/directory
```
The `--eval_model_dir` should specify the path to the directory containing the trained model weights.

### Visualization
During training, if the `--use_render` flag is set, an environment rendering window will display the agents' trajectories in real-time. You can also use the scripts in the `calculate/` directory to analyze and plot training data. For example:
```
$ python calculate/calculategraph.py
```
This will load the training log files from the `results/` directory and plot curves showing the agents' safety rate, success rate, time efficiency, and other metrics over the training time.

## Project Structure
- `mappo/`: Contains implementations of the MAPPO and RMAPPO algorithms in MetaDrive.
    - `algorithms/`: Contains implementations of the MAPPO and RMAPPO algorithms.
    - `calculate/`: Contains scripts for analyzing and visualizing training data.
    - `envs/`: Contains environment wrappers and definitions for RL training.
    - `runner/`: Contains training and evaluation logic.
    - `utils/`: Contains utility functions and buffer classes for recording training data.
    - `config.py`: Defines all training parameters.
    - `train/train.py`: Main script for training and evaluation.

## Configuration Parameters

Refer to [mappo/config.py](https://github.com/alstondu/MAPPO_AVs/blob/main/mappo/config.py) for detailed environment and agent configurations.

## Agent Policy Options

Setting ```agent_policy=IDMPolicy``` and ```agent_policy= ExpertPolicy``` in [train/train.py](https://github.com/alstondu/MAPPO_AVs/blob/main/mappo/train/train.py) to run IDM agents and PPO agents in the environment. For example, for using IDM policy:

```python
config_train = dict(
    ...
    agent_policy=IDMPolicy,
    ...
)

config_eval = dict(
    ...
    agent_policy=IDMPolicy,
    ...
)
```

## References

This project mainly references and uses the following algorithms, environments, and frameworks:

- MAPPO (Multi-Agent PPO): [Paper](https://arxiv.org/abs/2103.01955), [Code](https://github.com/marlbenchmark/on-policy)
- MetaDrive: [Website](https://decisionforce.github.io/metadrive/), [Code](https://github.com/decisionforce/metadrive)

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to express our gratitude to the following individuals and organizations for their contributions and support:

- The developers and contributors of the MetaDrive environment for providing a realistic and flexible simulation platform for autonomous driving research.
- The authors of the MAPPO algorithm for their groundbreaking work in multi-agent reinforcement learning.
- The open-source community for their valuable feedback, bug reports, and contributions to the project.

## Contact

If you have any questions, suggestions, or feedback regarding this project, please feel free to contact the project maintainers:

- Name ([@AlstonDu](https://github.com/AlstonDu))
- Email (ucab190@ucl.ac.uk)

We appreciate your interest in this project and look forward to your contributions and feedback!
