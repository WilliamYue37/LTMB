from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR
from ltmb.policies import Policy

# Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)
OBJECT_IDX = 0
COLOR_IDX = 1
STATE = 2

class ExpertMimicPolicy(Policy):
    def __init__(self):
        self.timestep = 0
        self.memory_associations = []
        self.color_to_timestep = {}
        self.color_to_action = {}
        self.next_unused_action = 0

    def select_action(self, obs):
        obs = obs['image']
        action = None
        
        self.memory_associations.append((2 * self.timestep, 2 * self.timestep)) # we need to pay attention to the current observation
        object, color = IDX_TO_OBJECT[obs[3, 3, OBJECT_IDX]], IDX_TO_COLOR[obs[3, 3, COLOR_IDX]]
        if object == 'key':
            if color not in self.color_to_timestep:
                self.color_to_timestep[color] = self.timestep
                action = self.next_unused_action
                self.color_to_action[color] = self.next_unused_action
                self.next_unused_action += 1
            else:
                action = self.color_to_action[color]
                # look at the first occurence of each key color
                for value in self.color_to_timestep.values():
                    self.memory_associations.append((2 * self.timestep, 2 * value)) # look at observation
                    self.memory_associations.append((2 * self.timestep, 2 * value + 1)) # look at action
        else:
            action = self.next_unused_action

        self.timestep += 1
        return action
    
    def get_memory_associations(self):
        return self.memory_associations