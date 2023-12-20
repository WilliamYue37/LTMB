class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def select_action(self):
        # Randomly select an action from the environment's action space
        return self.env.action_space.sample()