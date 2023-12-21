import gymnasium as gym
import pickle
import argparse
import ltmb
from ltmb.policies import ExpertHallwayPolicy

def collect_trajectories(env_name, seed, policy, num_trajectories):
    env = gym.make(env_name)
    trajectories = []
    avg_len, max_len = 0, 0

    for _ in range(num_trajectories):
        obs, info = env.reset(seed=seed)
        done = False
        trajectory = []

        while not done:
            action = policy.select_action(obs)
            trajectory.append((obs, action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        #assert info['success']
        trajectories.append(trajectory)
        avg_len += len(trajectory)
        max_len = max(max_len, len(trajectory))

    return trajectories, avg_len / num_trajectories, max_len

def save_trajectories(trajectories, filename):
    with open(filename, 'wb') as f:
        pickle.dump(trajectories, f)

def main():
    parser = argparse.ArgumentParser(description='Collect and save trajectories from a Gym environment.')
    parser.add_argument('--filename', type=str, required=True, help='Output file name for saved trajectories.')
    parser.add_argument('--runs', type=int, default=2, help='Number of trajectories to collect.')
    parser.add_argument('--env', type=str, required=True, help='Gym environment name.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    args = parser.parse_args()

    policy = ExpertHallwayPolicy()
    trajectories, avg_len, max_len = collect_trajectories(args.env, args.seed, policy, args.runs)
    print("Average length: ", avg_len)
    print("Max length: ", max_len)
    with open(args.filename, 'wb') as f:
        pickle.dump(trajectories, f)

if __name__ == '__main__':
    main()