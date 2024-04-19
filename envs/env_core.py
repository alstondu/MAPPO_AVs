import random
import time
import os

import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
import argparse
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import ManualControllableIDMPolicy
from collections import defaultdict
import numpy as np
import json

envs = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive
)

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self,args,config):
        self.args=args
        env_cls_name = args.env
        self.env= envs[env_cls_name](
            config
         )
        self.agent_num = self.args.num_agents
        self.obs_dim = list(self.env.observation_space.values())[0].shape[0]
        self.action_dim = list(self.env.action_space.values())[0].shape[0]
        self.action_space = list(self.env.action_space.values())
        self.observation_space=list(self.env.observation_space.values())
        self.need_reset=False
        self.agent_done_info = []
        self.metrics_data = defaultdict(list)
        self.reward_data = defaultdict(list)  # Add this line to store reward data
        self.agent_done_info_data = []
        self.epoch = 0
        self.time_step = 0
        self.temp_time_step = 0
        self.start_step = 0


    def reset(self):

        state = self.env.reset()
        sub_agent_obs=list(state[0].values())

        return sub_agent_obs              

    def step(self, actions):

        self.time_step += 1
        self.temp_time_step += 1
        sub_agent_obs,reward,done,truncateds,info = self.env.step({agent_id: action for agent_id,action in zip(self.env.vehicles.keys(),actions)})
        sub_agent_obs=list(sub_agent_obs.values())
        sub_agent_reward=list(reward.values())
        sub_agent_done = list(done.values())[:-1]
        sub_agent_info=list(info.values())
        # print(sub_agent_done)
        for agent_info in sub_agent_info:
            if any([
                agent_info.get("crash_vehicle", False),
                agent_info.get("crash_object", False),
                agent_info.get("crash_building", False),
                agent_info.get("crash_human", False),
                agent_info.get("crash_sidewalk", False),
                agent_info.get("out_of_road", False),
                agent_info.get("max_step", False),
                agent_info.get("crash", False),
                agent_info.get("arrive_dest", False)
            ]):
                self.agent_done_info.append(agent_info)

        new_agent_processed = False  # Flag to track if new agent's data is already used
        for agent_index, agent_done in enumerate(sub_agent_done):
            if agent_done:
                if len(sub_agent_done)>self.agent_num :
                    # if len(sub_agent_done)!=self.agent_num:
                    sub_agent_obs[agent_index] = sub_agent_obs[-1]
                    sub_agent_obs = np.delete(sub_agent_obs, -1, axis=0)
                    sub_agent_reward = np.delete(sub_agent_reward, -1)
                    sub_agent_done = np.delete(sub_agent_done, -1)
                    sub_agent_info = np.delete(sub_agent_info, -1)
                    new_agent_processed = True  # Mark that new agent's data has been used
                elif new_agent_processed and len(sub_agent_done)==self.agent_num:
                    # If another agent is done and the new agent's data is already used, skip processing
                    continue
                else:
                    # If the agent is done but hasn't reached its destination, reset the environment
                    # ref = my_env.reset()
                    break  # Assuming the entire environment is reset
        #if len(sub_agent_done)<self.agent_num:
            #print("len(sub_agent_done)<self.agent_num",len(sub_agent_done),len(sub_agent_obs))
        # print(reward)
        while len(sub_agent_done)<self.agent_num:
            sub_agent_done=np.append(sub_agent_done,False)  # If the agent is done but hasn't reached its destination, reset the environment
            sub_agent_reward=np.append(sub_agent_reward,0)
            # sub_agent_obs=np.append(sub_agent_obs,sub_agent_obs[-1])
            sub_agent_obs.append(sub_agent_obs[-1])
            sub_agent_info=np.append(sub_agent_info,sub_agent_info[-1])
            self.need_reset=False
            #print("processing:",len(sub_agent_done),len(sub_agent_obs))
            #print("Warning: Agent is done but hasn't reached its destination, reset the environment")

        if self.temp_time_step >= 5000:
            self.epoch += 1
            print(f"Epoch {self.epoch}: Information collection completed")
            #print("self.agent_done_info:", self.agent_done_info)
            total_agents = len(self.agent_done_info)
            crash_count = sum(1 for info in self.agent_done_info if info.get("crash", False))
            arrive_dest_count = sum(1 for info in self.agent_done_info if info.get("arrive_dest", False))
            total_episode_length = sum(info.get("episode_length", 0) for info in self.agent_done_info)
            success_episode_length = sum(info.get("episode_length", 0) for info in self.agent_done_info if info.get("arrive_dest", False))
            
            # Calculate Safety Rate
            safety_rate = (total_agents - crash_count) / total_agents if total_agents > 0 else 0
            
            # Calculate Success Rate
            success_rate = arrive_dest_count / total_agents if total_agents > 0 else 0
            
            # Calculate Time Efficiency
            if arrive_dest_count > 0:
                average_success_length = success_episode_length / arrive_dest_count
                time_efficiency = 1 - (average_success_length / 1000)
            else:
                time_efficiency = 0
            
            print(f"Safety Rate: {safety_rate:.2f}")
            print(f"Success Rate: {success_rate:.2f}")
            print(f"Time Efficiency: {time_efficiency:.2f}")
            num_vehicles = len(self.agent_done_info)
            print(f"Collected this round {num_vehicles} cars information")
            self.metrics_data[self.epoch] = [safety_rate, success_rate, time_efficiency]

            def convert_float32_to_float(obj):
                if isinstance(obj, dict):
                    return {k: convert_float32_to_float(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_float32_to_float(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_float32_to_float(item) for item in obj)
                elif isinstance(obj, np.float32):
                    return float(obj)
                else:
                    return obj
            converted_data = convert_float32_to_float(self.agent_done_info)                
            # Store agent_done_info data for the current epoch
            self.agent_done_info_data.append({
                            'start_step': self.start_step,
                            'end_step': self.time_step,
                            'agent_info': converted_data
                        })

            # Calculate average reward for the current epoch
            total_reward = sum(info.get("episode_reward", 0) for info in self.agent_done_info)
            avg_reward = total_reward / len(self.agent_done_info)
            self.reward_data[self.epoch].append(avg_reward)

            # Build output directory path
            output_dir = f"results/{self.args.env_name}/{self.args.scenario_name}/{self.args.algorithm_name}/check/{self.args.run_num}/logs"
            os.makedirs(output_dir, exist_ok=True)

            # Save reward data to file
            reward_file = os.path.join(output_dir, 'reward_data.json')
            with open(reward_file, 'w') as file:
                json.dump(dict(self.reward_data), file)

            # Save metrics_data to file
            metrics_file = os.path.join(output_dir, 'metrics_data.json')
            with open(metrics_file, 'w') as file:
                json.dump(dict(self.metrics_data), file)

            # Save agent_done_info_data to file
            agent_done_info_file = os.path.join(output_dir, 'agent_done_info_data.json')
            with open(agent_done_info_file, 'w') as file:
                json.dump(self.agent_done_info_data, file, indent=4)

            self.agent_done_info.clear()  # Clear the agent_done_info list
            self.temp_time_step = 0
            self.start_step = self.time_step
            self.need_reset = True



        # print(sub_agent_done, len(sub_agent_obs), len(sub_agent_reward), len(sub_agent_info))
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def close(self):
        self.env.close()

    def get_metrics_data(self):
        return self.metrics_data

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="intersection", choices=list(envs.keys()))
    parser.add_argument("--top_down", action="store_true",default=True)
    parser.add_argument("--num_agents", type=int,default=5)
    args = parser.parse_args()
    config=dict(
        horizon=200,
        use_render=True,
        crash_done= True,
        agent_policy=ManualControllableIDMPolicy,
        num_agents=args.num_agents,
        delay_done=False,
        vehicle_config=dict(
            lidar=dict(
                add_others_navi=False,
                num_others=4,
                distance=50,
                num_lasers=30,
            ),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12),

        )
    )
    my_env=EnvCore(args,config)
    my_env.reset()
    action=[np.zeros(2)+[0,1] for i in range(args.num_agents)]
    my_env.env.switch_to_third_person_view()  # Default is in Top-down vwwwwwwwwiew, we switch to Third-person view.
    while True:
        ref,r,done,info=my_env.step(action)
        print(len(done),len(ref))
        # print(my_env.env.episode_step,my_env.env.config["horizon"])
        if np.all(done) or my_env.env.episode_step >= my_env.env.config["horizon"]:
            ref = my_env.env.reset()  # reset the environment