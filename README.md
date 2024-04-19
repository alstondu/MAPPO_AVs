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
$ cd PATH/TO/MAPPO-AVs/mappo
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install -r requirements.txt
```

4. Replace the modified file in the MetaDrive lib:
```
$ cp torch_expert.py $CONDA_PREFIX/lib/python3.8/site-packages/metadrive/examples/ppo_expert/torch_expert.py
```

## Usage

### Training
To start training, use the `train/train.py` script. The script has the following important command-line arguments:
- `--scenario_name`: Specify the scenario for training, default is 'Intersection_MAPPO'.
- `--algorithm_name`: Specify the algorithm to use, supports 'mappo' and 'rmappo'.
- `--experiment_name`: Define the experiment name.
- `--num_agents`: Specify the number of agents in the environment.

For example, to start an experiment with 20 agents in the 'Intersection_MAPPO' scenario using the MAPPO algorithm, run:
```
$ cd PATH/TP/mappo
$ python train/train.py --scenario_name Intersection_MAPPO --algorithm_name mappo --experiment_name test_run --num_agents 20
```
Models and logs will be saved in the corresponding path under the `results/` directory.

### Evaluation
To perform evaluation using a trained model, use the `--eval` flag:
```
python mappo/train/train.py --eval --eval_model_dir /path/to/model/directory
```
The `--eval_model_dir` should specify the path to the directory containing the trained model weights.

### Visualization
During training, if the `--use_render` flag is set, an environment rendering window will display the agents' trajectories in real-time. You can also use the scripts in the `calculate/` directory to analyze and plot training data. For example:
```
python calculate/calculategraph.py
```
This will load the training log files from the `results/` directory and plot curves showing the agents' safety rate, success rate, efficiency, and other metrics over the training time.

## Project Structure
- `mappo/`: Contains implementations of the MAPPO and RMAPPO algorithms in MetaDrive.
    - `algorithms/`: Contains implementations of the MAPPO and RMAPPO algorithms.
    - `calculate/`: Contains scripts for analyzing and visualizing training data.
    - `envs/`: Contains environment wrappers and definitions for RL training.
    - `runner/`: Contains training and evaluation logic.
    - `utils/`: Contains utility functions and buffer classes for recording training data.
    - `config.py`: Defines all training parameters.
    - `train/train.py`: Main script for training and evaluation.
- `metadrive/`: Modified MetaDrive package for this project.

## Configuration Parameters

### Preparation Parameters
- `--algorithm_name`: Specifies the algorithm to use, either "rmappo" (Recurrent Multi-Agent Proximal Policy Optimization) or "mappo" (Multi-Agent Proximal Policy Optimization), default is "mappo".
- `--experiment_name`: Identifier to distinguish different experiments, default is "check".
- `--seed`: Sets the random seed for numpy and torch, default is 1.
- `--cuda`: Boolean, default is True, use GPU for training; otherwise, use CPU.
- `--cuda_deterministic`: Boolean, default is True, ensure random seed effectiveness; if set to False, it will bypass this functionality.
- `--n_training_threads`: Number of torch threads for training, default is 5.
- `--n_rollout_threads`: Number of parallel environments for training rollouts, default is 1.
- `--n_eval_rollout_threads`: Number of parallel environments for evaluation rollouts, default is 1.
- `--n_render_rollout_threads`: Number of parallel environments for rendering rollouts, default is 1.
- `--num_env_steps`: Number of environment steps for training, default is 10 million steps.
- `--user_name`: Used for wandb, specify the username for simple training data collection, default is "marl".

### Environment Parameters
- `--env_name`: Specifies the environment name, default is "COMP0124".
- `--use_obs_instead_of_state`: Boolean, default is False, use global state; if set to True, it will use concatenated observations.

### Replay Buffer Parameters
- `--episode_length`: Maximum length for any episode, default is 1000.

### Network Parameters
- `--share_policy`: Boolean, default is False, controls whether all agents share the same policy.
- `--use_centralized_V`: Boolean, default is True, use centralized value function estimation.
- `--stacked_frames`: Dimension of hidden layers for actor/critic networks, default is 1.
- `--use_stacked_frames`: Boolean, default is False, controls whether to use stacked frames.
- `--hidden_size`: Dimension of hidden layers for actor/critic networks, default is 256.
- `--layer_N`: Number of layers for actor/critic networks, default is 3.
- `--use_ReLU`: Boolean, default is False, controls whether to use ReLU activation function.
- `--use_popart`: Boolean, default is False, controls whether to use PopArt normalization for rewards.
- `--use_valuenorm`: Boolean, default is True, controls whether to use running mean and standard deviation to normalize rewards.
- `--use_feature_normalization`: Boolean, default is True, controls whether to apply layer normalization to inputs.
- `--use_orthogonal`: Boolean, default is True, controls whether to use orthogonal weight initialization and zero bias initialization.
- `--gain`: The gain of the last action layer, default is 0.01.

### Recurrent Policy Parameters
- `--use_naive_recurrent_policy`: Boolean, default is False, controls whether to use a simple recurrent policy.
- `--use_recurrent_policy`: Boolean, default is False, controls whether to use a recurrent policy.
- `--recurrent_N`: The number of recurrent layers, default is 1.
- `--data_chunk_length`: Length of data chunks for training recurrent policies, default is 10.

### Optimizer Parameters
- `--lr`: Learning rate, default is 5e-4.
- `--critic_lr`: Learning rate for the critic network, default is 5e-4.
- `--opti_eps`: Epsilon value for the RMSprop optimizer, default is 1e-5.
- `--weight_decay`: Weight decay coefficient, default is 0.

### PPO Parameters
- `--ppo_epoch`: Number of PPO epochs, default is 15.
- `--use_clipped_value_loss`: Boolean, default is True, clip value loss; if set, do not clip value loss.
- `--clip_param`: PPO clip parameter, default is 0.2.
- `--num_mini_batch`: Number of mini-batches for PPO, default is 1.
- `--entropy_coef`: Entropy coefficient, default is 0.01.
- `--value_loss_coef`: Value loss coefficient, default is 1.
- `--use_max_grad_norm`: Boolean, default is True, use the maximum gradient norm; if set, do not use.
- `--max_grad_norm`: Maximum gradient norm, default is 10.0.
- `--use_gae`: Boolean, default is True, use Generalized Advantage Estimation.
- `--gamma`: Discount factor for rewards, default is 0.99.
- `--gae_lambda`: Lambda parameter for GAE, default is 0.95.
- `--use_proper_time_limits`: Boolean, default is False, compute returns considering time limits.
- `--use_huber_loss`: Boolean, default is True, use Huber loss; if set, do not use Huber loss.
- `--use_value_active_masks`: Boolean, default is True, controls whether to mask useless data in value loss.
- `--use_policy_active_masks`: Boolean, default is True, controls whether to mask useless data in policy loss.
- `--huber_delta`: Coefficient for Huber loss, default is 10.0.

### Run Parameters
- `--use_linear_lr_decay`: Boolean, default is False, controls whether to use a linear decay strategy for learning rate.

### Save Parameters
- `--save_interval`: Time interval between two consecutive model saves, default is 1.

### Log Parameters
- `--log_interval`: Time interval between two consecutive log prints, default is 5.

### Evaluation Parameters
- `--use_eval`: Boolean, default is True, controls whether to start evaluation during training.
- `--eval_interval`: Time interval between two consecutive evaluation progress, default is 100.
- `--eval_episodes`: Number of episodes for a single evaluation, default is 10.

### Render Parameters
- `--save_gifs`: Boolean, default is False, controls whether to save rendered videos.
- `--use_render`: Boolean, default is False, controls whether to render the environment during training.
- `--use_render_eval`: Boolean, default is True, controls whether to render the environment during evaluation.
- `--render_episodes`: Number of episodes to render a given environment, default is 2.
- `--ifi`: Play interval of each rendered image in the saved video, default is 0.1.

### Pre-trained Parameters
- `--model_dir`: Sets the path to the pre-trained model, default is None.

### Environment Selection Parameters
- `--env`: Specifies the environment, can be any key from the envs dictionary, default is "roundabout".
- `--top_down`: Boolean, default is True, use a top-down view.
- `--num_agents`: Number of agents, default is 2.
- `--random_traffic`: Boolean, default is True, randomly place other vehicles on the road.
- `--human_vehicle`: Boolean, default is True, presence of other vehicles on the road.
- `--traffic_density`: Dictionary type, default is the density variable, specifies traffic density.
- `--obs_num_others`: Number of other agents observed by each agent, default is 4.
- `--show_navi`: Boolean, default is True, display navigation marks.
- `--show_dest`: Boolean, default is True, display destination marks.

## Agent Policy Options

In the MetaDrive environment, `agent_policy` can be set to different traffic vehicle behavior policies. Here are some common options:

1. `IDMPolicy`: Intelligent Driver Model policy, which is a policy based on the vehicle following model. It adjusts the vehicle's acceleration and braking according to the distance and speed difference from the leading vehicle.

2. `ManualControllableIDMPolicy`: Manually controllable IDM policy, allowing human takeover of vehicle control.

3. `ControlledIDMPolicy`: Controllable IDM policy, similar to `ManualControllableIDMPolicy`, but controls the vehicle through a program instead of human control.

4. `LinearPolicy`: Linear policy, which linearly calculates the vehicle's steering control based on the deviation from the road centerline and the yaw angle error.

5. `WaymoIDMPolicy`: Waymo's IDM policy, which is a tuned IDM policy used to simulate the behavior of Waymo's self-driving cars.

6. `EndToEndPolicy`: End-to-end policy, which directly outputs vehicle control commands based on observations through a neural network.

7. `MacroIDMPolicy`: Macro IDM policy, which considers the macro behavior of vehicles in the road network, such as lane changes and overtaking.

You can choose a suitable policy according to your needs by replacing the value of `agent_policy`. For example, if you want to use the end-to-end policy, you can set:

```python
config_train = dict(
    ...
    agent_policy=EndToEndPolicy,
    ...
)

config_eval = dict(
    ...
    agent_policy=EndToEndPolicy,
    ...
)
```

Please note that different policies may require different observation information and control outputs, so after changing `agent_policy`, ensure that the observation space and action space are compatible with the selected policy.

## Background Traffic

In the MetaDrive environment, in addition to the vehicles controlled by the agents, you can add other traffic participants, i.e., background traffic. These vehicles are automatically controlled by the environment to simulate real traffic scenarios.

By setting the `num_agents` parameter, you can control the number of agents, i.e., the number of vehicles controlled by the reinforcement learning algorithm.

The number and properties of other vehicles in the environment can be controlled by the following parameters:

1. `random_traffic`: Boolean value, indicating whether to generate random traffic flow in the environment. If set to `True`, other vehicles will be randomly generated in the environment.

2. `traffic_density`: Traffic flow density, controlling the number of vehicles generated. The larger the value, the more vehicles generated and the more congested the traffic.

3. `vehicle_config`: Dictionary used to configure vehicle properties, including sensors, navigation information display, vehicle color, etc. These configurations will be applied to all vehicles, including agent-controlled vehicles and background vehicles.

For example, if you want to add random traffic flow to the environment and control the traffic density, you can set:

```python
config_train = dict(
    ...
    num_agents=5,  # Set the number of agents to 5
    random_traffic=True,  # Enable random traffic flow
    traffic_density=0.1,  # Set the traffic density to 0.1
    ...
)

config_eval = dict(
    ...
    num_agents=5,
    random_traffic=True,
    traffic_density=0.1,
    ...
)
```

This way, in both the training and evaluation environments, random traffic flow will be generated with a traffic density of 0.1. At the same time, there will be 5 vehicles controlled by the agents.

The behavior of other vehicles in the environment is controlled by the `IDMPolicy`, which is an intelligent driving model that mimics real traffic. You can also change the behavior policy of these background vehicles by modifying `agent_policy`.

## Number of Background Vehicles

In the MetaDrive environment, you can control the number of background vehicles controlled by the IDM policy by setting the `num_background_vehicles` parameter. This parameter directly specifies the exact number of background vehicles in the environment.

Here's an example showing how to set `num_background_vehicles` in the configuration:

```python
config_train = dict(
    ...
    num_agents=5,  # Set the number of agents to 5
    random_traffic=True,  # Enable random traffic flow
    traffic_density=0.1,  # Set the traffic density to 0.1
    num_background_vehicles=20,  # Set the number of background vehicles to 20
    ...
)

config_eval = dict(
    ...
    num_agents=5,
    random_traffic=True,
    traffic_density=0.1,
    num_background_vehicles=20,
    ...
)
```

In this example, we set `num_background_vehicles` to 20, which means that in the training and evaluation environments, in addition to the 5 vehicles controlled by the agents, there will be 20 vehicles controlled by the IDM policy.

Please note that the `traffic_density` parameter and the `num_background_vehicles` parameter have a certain degree of mutual exclusivity. If both parameters are set simultaneously, the setting of `num_background_vehicles` will override the effect of `traffic_density`. Therefore, it is recommended to set only one of these parameters to control the background traffic.

Also, make sure not to set the value of `num_background_vehicles` too large to avoid having too many vehicles in the environment, which may affect the performance of training and evaluation. The specific value needs to be reasonably set according to the size and complexity of the scenario.

## Episode Termination Conditions

In the configuration of the MetaDrive environment, `crash_done` and `delay_done` are two parameters related to the episode termination conditions.

1. `crash_done`:
   - When `crash_done` is set to `True`, if the vehicle controlled by the agent collides (with other vehicles, road boundaries, or obstacles), the current episode will immediately end, and control will be returned to the reinforcement learning algorithm.
   - If `crash_done` is set to `False`, the episode will not immediately end after a vehicle collision, and the vehicle will continue driving. However, collisions usually have a negative impact on the reward function, so the algorithm will still try to learn to avoid collisions.

2. `delay_done`:
   - When `delay_done` is set to `True`, if the vehicle controlled by the agent fails to reach the target position within a certain time, the current episode will end prematurely. This time limit is controlled by the `horizon` parameter, which represents the maximum number of steps or duration of an episode.
   - If `delay_done` is set to `False`, the episode will not end prematurely due to the vehicle taking too long to reach the target position. The vehicle will continue driving until it reaches the step or time limit specified by `horizon`.

In your configuration, `crash_done` is set to `True`, and `delay_done` is set to `False`, which means:
- If a vehicle collides, the current episode will immediately end;
- If a vehicle fails to reach the target within the specified time (controlled by the `horizon` parameter), the episode will not end prematurely, and the vehicle will continue driving until it reaches the limit of `horizon`.

The purpose of this configuration is to encourage the agent to learn to avoid collisions while giving the agent sufficient time to reach the target position. You can adjust the settings of `crash_done` and `delay_done`, as well as the value of the `horizon` parameter, based on your specific training goals and requirements to obtain the desired training effect.

## Background Traffic Removal

Even if you comment out `agent_policy=ManualControllableIDMPolicy`, there will still be vehicles running in the environment that are not controlled by the MAPPO algorithm. This is because the MetaDrive environment generates a certain number of background traffic by default to simulate real traffic scenarios.

These background vehicles are controlled by the environment's built-in IDM (Intelligent Driver Model) policy, which is independent of the vehicles controlled by the agents. Their behavior is intended to mimic human drivers in real traffic.

If you want to completely remove the background traffic from the environment, you can set `random_traffic` to `False` and `traffic_density` to 0. For example:

```python
config_train = dict(
    ...
    random_traffic=False,  # Disable random traffic flow
    traffic_density=0,  # Set traffic density to 0
    ...
)

config_eval = dict(
    ...
    random_traffic=False,
    traffic_density=0,
    ...
)
```

With this setting, no background traffic will be generated in the environment, and all vehicles will be controlled by the MAPPO algorithm.

Please note that completely removing background traffic may affect the training effectiveness of the algorithm, as the agents will not learn how to deal with the presence of other vehicles. In practical applications, it is usually recommended to retain a certain amount of background traffic to allow the agents to learn to drive in more realistic traffic environments. You can control the amount of background traffic by adjusting the `traffic_density` or `num_background_vehicles` parameters to achieve a balance between training effectiveness and environment realism.

## References

This project mainly references and uses the following algorithms, environments, and frameworks:

- MAPPO (Multi-Agent PPO): [Paper](https://arxiv.org/abs/2103.01955), [Code](https://github.com/marlbenchmark/on-policy)
- MetaDrive: [Website](https://decisionforce.github.io/metadrive/), [Code](https://github.com/decisionforce/metadrive)

## Maintainers

Currently, this project is being developed and maintained by [@AlstonDu](https://github.com/AlstonDu). If you have any questions or suggestions for improvement, please feel free to open an issue or submit a pull request.

## Troubleshooting

If you encounter any issues or errors while setting up or running the project, please refer to the following troubleshooting tips:

1. **Installation Issues**: Make sure you have followed the installation steps correctly and have installed all the required dependencies. Double-check the versions of the packages and ensure they are compatible with your Python environment.

2. **Environment Rendering**: If you encounter problems with environment rendering, such as the environment window not displaying or crashing, try updating your graphics drivers and ensure that your system meets the minimum requirements for running the MetaDrive environment.

3. **Training Performance**: If the training process is slow or consumes too much memory, you can try reducing the number of parallel environments (`n_rollout_threads`) or adjusting the batch size and number of training steps. Additionally, make sure you have sufficient GPU memory if you are using a GPU for training.

4. **Evaluation Results**: If the evaluation results are not as expected, double-check your configuration settings and ensure that the trained model is being loaded correctly. You can also try increasing the number of evaluation episodes (`eval_episodes`) to get a more accurate assessment of the model's performance.

5. **Reproducibility**: To ensure reproducibility of results, make sure to set the random seed (`seed`) to a fixed value across different runs. This will help in obtaining consistent results and facilitates comparisons between different experiments.

If you encounter any other issues or have specific questions, please feel free to reach out to the project maintainers or open an issue on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for both commercial and non-commercial purposes. However, please note that the project maintainers and contributors are not liable for any damages or issues arising from the use of this software.

## Acknowledgments

We would like to express our gratitude to the following individuals and organizations for their contributions and support:

- The developers and contributors of the MetaDrive environment for providing a realistic and flexible simulation platform for autonomous driving research.
- The authors of the MAPPO algorithm for their groundbreaking work in multi-agent reinforcement learning.
- The open-source community for their valuable feedback, bug reports, and contributions to the project.

## Contact

If you have any questions, suggestions, or feedback regarding this project, please feel free to contact the project maintainers:

- Your Name ([@AlstonDu](https://github.com/AlstonDu))
- Your Email (ucab190@ucl.ac.uk)

We appreciate your interest in this project and look forward to your contributions and feedback!
