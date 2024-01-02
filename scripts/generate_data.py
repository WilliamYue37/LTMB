import gymnasium as gym
import pickle
import argparse
import ltmb
import random
from ltmb.policies import Policy, ExpertHallwayPolicy, ExpertMimicPolicy, ExpertCountingPolicy
from typing import Type

def record_video(env_name: str, expert: Type[Policy], filename: str, length: int):
    policy = expert()
    env = gym.make(env_name, render_mode='rgb_array', length=length)
    env = gym.wrappers.RecordVideo(env, filename + '_recording')
    obs, info = env.reset(seed=random.randint(0, 10**9))
    done = False

    while not done:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    assert info['success']
    env.close()

def collect_trajectories(env_name: str, expert: Type[Policy], num_trajectories: int, length: int):
    env = gym.make(env_name, length=length)
    trajectories = []
    avg_len, max_len = 0, 0

    for _ in range(num_trajectories):
        policy = expert() # policy is not markovian and must be reinitialized for each trajectory
        obs, info = env.reset(seed=random.randint(0, 10**9))
        done = False
        trajectory = []

        while not done:
            action = policy.select_action(obs)
            trajectory.append((obs, action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        assert info['success']
        trajectories.append((trajectory, policy.get_memory_associations()))
        avg_len += len(trajectory)
        max_len = max(max_len, len(trajectory))

    env.close()
    return trajectories, avg_len / num_trajectories, max_len

def main():
    parser = argparse.ArgumentParser(description='Collect and save trajectories from a Gym environment.')
    parser.add_argument('--filename', type=str, required=True, help='Output file name for saved trajectories. (*.pkl)')
    parser.add_argument('--runs', type=int, default=2, help='Number of trajectories to collect.')
    parser.add_argument('--env', type=str, required=True, choices=['LTMB-Hallway-v0', 'LTMB-Mimic-v0', 'LTMB-Counting-v0'], help='Gym environment name.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--length', type=int, default=10, help='Length of the task.')
    parser.add_argument('--record', action='store_true', help='Record a video of the expert policy.')
    args = parser.parse_args()

    random.seed(args.seed)

    expert = None
    if args.env == 'LTMB-Hallway-v0':
        expert = ExpertHallwayPolicy
    elif args.env == 'LTMB-Mimic-v0':
        expert = ExpertMimicPolicy
    elif args.env == 'LTMB-Counting-v0':
        expert = ExpertCountingPolicy
    assert expert is not None

    trajectories, avg_len, max_len = collect_trajectories(args.env, expert, args.runs, args.length)
    print("Average length: ", avg_len)
    print("Max length: ", max_len)
    with open(args.filename, 'wb') as f:
        pickle.dump(trajectories, f)
    
    if args.record:
        record_video(args.env, expert, args.filename, args.length)

if __name__ == '__main__':
    main()