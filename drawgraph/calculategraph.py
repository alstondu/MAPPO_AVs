import json
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(file_path, start_step, end_step):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    total_agents = 0
    crash_count = 0
    arrive_dest_count = 0
    total_episode_length = 0
    success_episode_length = 0
    total_episode_reward = 0

    for item in data:
        if start_step <= item['start_step'] < end_step:
            for agent_info in item['agent_info']:
                total_agents += 1
                if agent_info['crash']:
                    crash_count += 1
                if agent_info['arrive_dest']:
                    arrive_dest_count += 1
                    success_episode_length += agent_info['episode_length']
                total_episode_length += agent_info['episode_length']
                total_episode_reward += agent_info['episode_reward']

    safety = 1 - (crash_count / total_agents)
    success_rate = arrive_dest_count / total_agents
    efficiency = 1 - ((success_episode_length / arrive_dest_count) / 5000) if arrive_dest_count > 0 else 0
    avg_episode_reward = total_episode_reward / total_agents

    return safety, success_rate, efficiency, avg_episode_reward

def main():
    # Specified file paths and step range
    file_configs = [
        {'path': 'mappo256.json', 'step_range': range(0, 1000000, 50000), 'label': 'MAPPO256'},
        {'path': 'mappo64.json', 'step_range': range(0, 1000000, 50000), 'label': 'MAPPO64'},
        {'path': 'rmappo256.json', 'step_range': range(0, 1000000, 50000), 'label': 'RMAPPO256'},
        {'path': 'rmappo64.json', 'step_range': range(0, 1000000, 50000), 'label': 'RAMPPO64'} 
    ]

    # Calculate metrics for each file and step range
    metrics_data = {}
    for config in file_configs:
        metrics_data[config['label']] = []
        for start_step in config['step_range']:
            end_step = start_step + 50000
            safety, success_rate, efficiency, avg_episode_reward = calculate_metrics(config['path'], start_step, end_step)
            metrics_data[config['label']].append([start_step, safety, success_rate, efficiency, avg_episode_reward])

    # Plot metrics curves
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, metric_name in enumerate(['Safety Rate', 'Success Rate', 'Efficiency Rate', 'Avg. Episode Reward']):
        for label, data in metrics_data.items():
            steps = [row[0] for row in data]
            metric_values = [row[i+1] for row in data]
            axs[i].plot(steps, metric_values, linewidth=2, linestyle='solid', label=label)
        axs[i].set_xlabel('Training Step')
        axs[i].set_ylabel(metric_name)
        axs[i].grid(True)
        axs[i].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()