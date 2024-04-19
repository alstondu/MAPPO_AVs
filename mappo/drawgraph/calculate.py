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

    safety = 1 - (crash_count / total_agents)
    success_rate = arrive_dest_count / total_agents
    efficiency = 1 - ((success_episode_length / arrive_dest_count) / 1000) if arrive_dest_count > 0 else 0

    return safety, success_rate, efficiency, total_agents, arrive_dest_count

def main():
    # Specified file paths and start/end steps
    file_configs = [
        {'path': 'idm.json', 'start_step': 0, 'end_step': 5000, 'label': 'IDM'},
        {'path': 'ppo.json', 'start_step': 500000, 'end_step': 505000, 'label': 'PPO'},
        {'path': 'mappo64.json', 'start_step': 775000, 'end_step': 780000, 'label': 'MAPPO64'}
    ]

    # Calculate metrics for each file
    metrics = []
    for config in file_configs:
        safety, success_rate, efficiency, total_agents, arrive_dest_count = calculate_metrics(config['path'], config['start_step'], config['end_step'])
        metrics.append([safety, success_rate, efficiency])
        print(f"{config['label']}:")
        print(f"Total Agents: {total_agents}")
        print(f"Arrived Destinations: {arrive_dest_count}")
        print(f"Safety: {safety:.2f}")
        print(f"Success Rate: {success_rate:.2f}")
        print(f"Efficiency: {efficiency:.2f}")
        print()

    # Plot radar chart
    labels = ['Safety Rate', 'Success Rate', 'Time Efficiency']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    for i, metric in enumerate(metrics):
        values = metric + metric[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=file_configs[i]['label'])
        ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.show()

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()