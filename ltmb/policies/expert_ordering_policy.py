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
            for pos in [(2, 3), (4, 3), (3, 2), (3, 4)]:
                object, color = IDX_TO_OBJECT[obs[pos[0], pos[1], OBJECT_IDX]], IDX_TO_COLOR[obs[pos[0], pos[1], COLOR_IDX]]
                choices.append((object, color))

            min_index = min(self.permutation.index(choice) for choice in choices)
            for i in range(4):
                idx = self.permutation.index(choices[i])
                self.memory_associations.append((2 * self.timestep, 2 * idx)) # we need to pay attention to the object
            
            action = None
            if self.permutation.index(choices[0]) == min_index:
                action = Actions.left
            elif self.permutation.index(choices[1]) == min_index:
                action = Actions.right
            elif self.permutation.index(choices[2]) == min_index:
                action = Actions.forward
            elif self.permutation.index(choices[3]) == min_index:
                action = Actions.toggle
            assert action is not None

        self.timestep += 1
        return action
    
    def get_memory_associations(self):
        return self.memory_associations