from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_OBJECT

# Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)
OBJECT_IDX = 0
COLOR_IDX = 1
STATE = 2

class ExpertHallwayPolicy:
    def __init__(self):
        self.timestep = 0
        self.memory_associations = []
        self.target_object = None
        self.target_color = None

    def select_action(self, obs):
        obs = obs['image']
        action = None
        
        if self.timestep == 0: # observe starting room at start of episode
            action = Actions.left
        elif self.timestep == 1: # turn back to face the hallway after observing the starting room
            for i in range(obs.shape[0]):
                for j in range(obs.shape[1]):
                    if IDX_TO_OBJECT[obs[i, j, OBJECT_IDX]] in ['key', 'ball', 'box']:
                        print('found object')
            action = Actions.right
        elif self.timestep >= 2: # move forward into the hallway
            action = Actions.forward
        
        self.timestep += 1
        return action
    
    def get_memory_associations(self):
        return self.memory_associations