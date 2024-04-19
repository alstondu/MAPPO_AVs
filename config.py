import argparse
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)

def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.

    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.

    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer.

    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies.
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards.
        --use_valuenorm
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)

    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.
        --huber_delta <float>
            coefficient of huber loss.

    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)

    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate

    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.

    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.

    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="rmappo", choices=["rmappo", "mappo"])

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=4,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollouts",
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollouts",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=5e6,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )

    # env parameters
    parser.add_argument("--env_name", type=str, default="MyEnv", help="specify the name of environment")
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=5000, help="Max length for any episode")

    # network parameters
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=True,
        help="Whether agent share the same policy",
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--layer_N",
        type=int,
        default=3,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument("--use_ReLU", action="store_false", default=False, help="Whether to use ReLU")
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=True,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=False,
        help="by default True, use running mean and std to normalize rewards.",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=True,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=True,
        help="use a recurrent policy",
    )
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)")
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=1e-3,
        help="critic learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15, help="number of ppo epochs (default: 15)")
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=True,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=True,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice log printing.",
    )

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=50,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )

    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=True,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )

    parser.add_argument(
        "--use_render_eval",
        action="store_true",
        default=True,
        help="by default, do not render the env during evaling. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )


    parser.add_argument(
        "--render_episodes",
        type=int,
        default=2,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )


    density = dict(
        roundabout=0,
        intersection=0,
        tollgate=0,
        bottleneck=0,
        parkinglot=0,
        pgma=0
    )
    envs = dict(
        roundabout=MultiAgentRoundaboutEnv,
        intersection=MultiAgentIntersectionEnv,
        tollgate=MultiAgentTollgateEnv,
        bottleneck=MultiAgentBottleneckEnv,
        parkinglot=MultiAgentParkingLotEnv,
        pgma=MultiAgentMetaDrive,
    )
    parser.add_argument("--env", type=str, default="roundabout", choices=list(envs.keys()))
    parser.add_argument("--top_down", default=True,action="store_true")
    parser.add_argument("--num_agents", type=int,default=20)
    parser.add_argument(
        "--random_traffic",
        type=bool,
        default=False,
        help="by default True, other human vehicle randomly on road.",
    )
    parser.add_argument(
        "--human_vehicle",
        type=bool,
        default=False,
        help="by default True, other human vehicle on road.",
    )
    parser.add_argument("--traffic_density", type=dict,default=density)
    parser.add_argument("--obs_num_others", type=int, default=4, help="the number of agent's observation")
    parser.add_argument("--show_navi", type=bool, default=True, help="whether show navi mark")
    parser.add_argument("--show_dest", type=bool, default=True, help="whether show navi mark")
    parser.add_argument("--run_num", type=int, default=None, help="Number of the current run")


    return parser




if __name__=="__main__":
    # Prepare parameters
    algorithm_name = "mappo"  # The algorithm to use
    experiment_name = "check"  # Identifier for the experiment
    seed = 1  # Random seed for numpy/torch
    cuda = True  # Use GPU for training (True) or CPU (False)
    cuda_deterministic = True  # Ensure random seed effectiveness (True) or bypass (False)
    n_training_threads = 2  # Number of training threads
    n_rollout_threads = 1  # Number of parallel envs for training rollouts
    n_eval_rollout_threads = 1  # Number of parallel envs for evaluating rollouts
    n_render_rollout_threads = 1  # Number of parallel envs for rendering rollouts
    num_env_steps = 10000000  # Number of environment steps to train
    user_name = "marl"  # User's name for collecting training data

    # Env parameters
    env_name = "MyEn"  # Name of the environment
    use_obs_instead_of_state = False  # Use global state (False) or concatenated observations (True)

    # Replay Buffer parameters
    episode_length = 200  # Maximum episode length

    # Network parameters
    share_policy = False  # Agents share the same policy (False) or not (True)
    use_centralized_V = True  # Use centralized V function (True) or not (False)
    stacked_frames = 1  # Number of stacked frames
    use_stacked_frames = False  # Use stacked frames (True) or not (False)
    hidden_size = 64  # Dimension of hidden layers for actor/critic networks
    layer_N = 1  # Number of layers for actor/critic networks
    use_ReLU = False  # Use ReLU (True) or not (False)
    use_popart = False  # Use PopArt to normalize rewards (False)
    use_valuenorm = True  # Use running mean and std to normalize rewards (True)
    use_feature_normalization = True  # Apply layernorm to normalize inputs (True)
    use_orthogonal = True  # Use Orthogonal initialization for weights and 0 initialization for biases (True)
    gain = 0.01  # Gain for the last action layer

    # Recurrent parameters
    use_naive_recurrent_policy = False  # Use a naive recurrent policy (True) or not (False)
    use_recurrent_policy = False  # Use a recurrent policy (True) or not (False)
    recurrent_N = 1  # Number of recurrent layers
    data_chunk_length = 10  # Time length of chunks used to train a recurrent policy

    # Optimizer parameters
    lr = 0.0005  # Learning rate
    critic_lr = 0.0005  # Critic learning rate
    opti_eps = 1e-05  # RMSprop optimizer epsilon
    weight_decay = 0  # Coefficient of weight decay

    # PPO parameters
    ppo_epoch = 15  # Number of PPO epochs
    use_clipped_value_loss = False  # Clip value loss (False) or not (True)
    clip_param = 0.2  # PPO clip parameter
    num_mini_batch = 1  # Number of batches for PPO
    entropy_coef = 0.01  # Entropy term coefficient
    value_loss_coef = 1  # Value loss coefficient
    use_max_grad_norm = False  # Use max norm of gradients (False) or not (True)
    max_grad_norm = 10.0  # Max norm of gradients
    use_gae = False  # Use generalized advantage estimation (False) or not (True)
    gamma = 0.99  # Discount factor for rewards
    gae_lambda = 0.95  # GAE lambda parameter
    use_proper_time_limits = True  # Compute returns taking into account time limits (True) or not (False)
    use_huber_loss = False  # Use huber loss (False) or not (True)
    use_value_active_masks = False  # Mask useless data in value loss (True) or not (False)
    use_policy_active_masks = False  # Mask useless data in policy loss (True) or not (False)
    huber_delta = 10.0  # Coefficient of huber loss

    # PPG parameters
    aux_epoch = 4  # Number of auxiliary epochs
    clone_coef = 0.01  # Clone term coefficient

    # Run parameters
    use_linear_lr_decay = False  # Apply linear decay to learning rate (True) or not (False)

    # Save & Log parameters
    save_interval = 1  # Time duration between continuous model saving
    log_interval = 5  # Time duration between continuous log printing

    # Eval parameters
    use_eval = True  # Start evaluation alongside with training (True) or not (False)
    eval_interval = 3  # Time duration between continuous evaluation progress
    eval_episodes = 32  # Number of episodes for a single evaluation

    # Render parameters
    save_gifs = False  # Save render video (True) or not (False)
    use_render = False  # Render the environment during training (True) or not (False)
    render_episodes = 5  # Number of episodes to render a given env
    ifi = 0.1  # Play interval of each rendered image in saved video

    # Pretrained parameters
    model_dir = None  # Path to pretrained model

    # Environment selection
    env = ""  # Name of the environment
    top_down = False  # Use top-down view (True) or not (False)
    num_agent = 2  # Number of agents

    #        num_updates = self.ppo_epoch * self.num_mini_batch