import random

class RandomPolicy:
    def select_action(self, obs):
        # Randomly select an action from the environment's action space
        return random.randint(0, 6)