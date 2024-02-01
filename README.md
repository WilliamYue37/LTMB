# LTMB
**Long-term Memory Benchmark for Sequential Decision Making**

![Figure 1](images/LTMB.png)

# Installtion
```shell
pip install -e .
```

# Getting Started

Run the following commands to play each task.

```shell
play_hallway
play_ordering
play_counting
```

# Expert Policies

Expert policies for each task are located in [./ltmb/policies/](./ltmb/policies). All expert policies have a `select_action(obs)` and `get_memory_associations()` method.

# Dataset Collection

LTMB datasets can be generated using the script [./scripts/generate\_data.py](./scripts/generate\_data.py). Run the following command to view the full list of options:

```shell
python ./scripts/generate\_data.py --help
```

The LTMB datasets used in the AttentionTuner paper were generated with the following commands:

```shell
python ./scripts/generate\_data.py --filename hallway.pkl --runs 4000 --env LTMB-Hallway-v0 --seed 0 --length 30
python ./scripts/generate\_data.py --filename ordering.pkl --runs 5000 --env LTMB-Ordering-v0 --seed 0 --length 50
python ./scripts/generate\_data.py --filename counting.pkl --runs 10000 --env LTMB-Counting-v0 --seed 0 --length 20
```

# Citation
If you find **LTMB** to be useful in your own research, please consider citing our paper:

# Acknowledgements
LTMB is built on top of the [Minigrid](https://github.com/Farama-Foundation/Minigrid) environment.
