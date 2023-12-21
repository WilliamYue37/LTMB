import gymnasium as gym
import pickle
import argparse
import ltmb
from ltmb.policies import RandomPolicy

def collect_trajectories(env_name, policy, num_trajectories):
    env = gym.make(env_name)
    trajectories = []

    for _ in range(num_trajectories):
        obs, info = env.reset()
        done = False
        trajectory = []

        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            trajectory.append((obs, action))
            obs = obs

        trajectories.append(trajectory)

    return trajectories

def save_trajectories(trajectories, filename):
    with open(filename, 'wb') as f:
        pickle.dump(trajectories, f)

def main():
    parser = argparse.ArgumentParser(description='Collect and save trajectories from a Gym environment.')
    parser.add_argument('--filename', type=str, required=True, help='Output file name for saved trajectories.')
    parser.add_argument('--runs', type=int, default=2, help='Number of trajectories to collect.')
    parser.add_argument('--env', type=str, required=True, help='Gym environment name.')
    args = parser.parse_args()

    policy = RandomPolicy(gym.make(args.env))
    trajectories = collect_trajectories(args.env, policy, args.runs)
    with open(args.filename, 'wb') as f:
        pickle.dump(trajectories, f)

if __name__ == '__main__':
    main()