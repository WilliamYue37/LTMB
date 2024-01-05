from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR
from queue import Queue
from ltmb.policies import Policy

# Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)
OBJECT_IDX = 0
COLOR_IDX = 1
STATE = 2

class ExpertHallwayPolicy(Policy):
    def __init__(self):
        self.timestep = 0
        self.memory_associations = []
        self.target_object = None
        self.target_color = None
        self.action_queue = Queue()

    def select_action(self, obs):
        obs = obs['image']
        action = None
        
        self.memory_associations.append((2 * self.timestep, 2 * self.timestep)) # we need to pay attention to the current observation
        if self.action_queue.qsize() > 0:
            action = self.action_queue.get()
        elif self.timestep == 0: # observe starting room at start of episode
            action = Actions.left # turn to look for target object
        elif self.timestep == 1: # turn back to face the hallway after observing the starting room
            assert IDX_TO_OBJECT[obs[2, 5, OBJECT_IDX]] in ['key', 'ball', 'box'] # the target object should be here
            self.target_object = IDX_TO_OBJECT[obs[2, 5, OBJECT_IDX]]
            self.target_color = IDX_TO_COLOR[obs[2, 5, COLOR_IDX]]
            action = Actions.right # turn back to face the hallway
        elif self.timestep >= 2: # move forward into the hallway
            # check if the object in the vertical hallway is the target object
            if IDX_TO_OBJECT[obs[1, 6, OBJECT_IDX]] == self.target_object and IDX_TO_COLOR[obs[1, 6, COLOR_IDX]] == self.target_color:
                action = Actions.left # turn to face the door
                self.action_queue.put(Actions.toggle) # toggle the door
                self.action_queue.put(Actions.forward) # move forward through the door
            elif IDX_TO_OBJECT[obs[5, 6, OBJECT_IDX]] == self.target_object and IDX_TO_COLOR[obs[5, 6, COLOR_IDX]] == self.target_color:
                action = Actions.right # turn to face the door
                self.action_queue.put(Actions.toggle) # toggle the door
                self.action_queue.put(Actions.forward) # move forward through the door
            else:
                action = Actions.forward

            # add memory association if we are next to a door
            if IDX_TO_OBJECT[obs[2, 6, OBJECT_IDX]] == 'door' or IDX_TO_OBJECT[obs[4, 6, OBJECT_IDX]] == 'door':
                self.memory_associations.append((2 * self.timestep, 2 * 1)) # multiply by 2 because observations are at even indicies and actions are at odd indicies
        
        self.timestep += 1
        return action
    
    def get_memory_associations(self):
        return self.memory_associations