import random
from ltmb.policies.policy import Policy

class RandomPolicy(Policy):
    def select_action(self, obs):
        # Randomly select an action from the environment's action space
        return random.randint(0, 6)

    def get_memory_associations(self):
        return []