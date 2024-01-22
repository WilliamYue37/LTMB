from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR
from ltmb.policies import Policy

# Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)
OBJECT_IDX = 0
COLOR_IDX = 1
STATE = 2

class ExpertOrderingPolicy(Policy):
    def __init__(self):
        self.timestep = 0
        self.memory_associations = []
        self.permutation = []

    def select_action(self, obs):
        obs = obs['image']
        action = None
        
        self.memory_associations.append((2 * self.timestep, 2 * self.timestep)) # we need to pay attention to the current observation
        action = Actions.forward
        if self.timestep < 18:
            object, color = IDX_TO_OBJECT[obs[3, 3, OBJECT_IDX]], IDX_TO_COLOR[obs[3, 3, COLOR_IDX]]
            self.permutation.append((object, color))
        else:
            choices = []
            for i in [2, 4]:
                object, color = IDX_TO_OBJECT[obs[i, 3, OBJECT_IDX]], IDX_TO_COLOR[obs[i, 3, COLOR_IDX]]
                choices.append((object, color))
            left_idx, right_idx = self.permutation.index(choices[0]), self.permutation.index(choices[1])
            self.memory_associations.append((2 * self.timestep, 2 * left_idx)) # we need to pay attention the left object
            self.memory_associations.append((2 * self.timestep, 2 * right_idx)) # we need to pay attention the right object
            if left_idx < right_idx:
                action = Actions.left
            else:
                action = Actions.right

        self.timestep += 1
        return action
    
    def get_memory_associations(self):
        return self.memory_associations