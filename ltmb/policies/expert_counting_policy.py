from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR
from ltmb.policies import Policy
from collections import defaultdict
from queue import Queue

# Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)
OBJECT_IDX = 0
COLOR_IDX = 1
STATE = 2

class ExpertCountingPolicy(Policy):
    def __init__(self):
        self.timestep = 0
        self.memory_associations = []
        self.objects_seen = defaultdict(list) # maps object to list of timesteps
        self.action_queue = Queue()

    def _get_object(self, obs, x, y):
        return IDX_TO_OBJECT[obs[x, y, OBJECT_IDX]]
    
    def _get_color(self, obs, x, y):
        return IDX_TO_COLOR[obs[x, y, COLOR_IDX]]

    def select_action(self, obs):
        obs = obs['image']
        action = None

        if self.action_queue.qsize() > 0:
            action = self.action_queue.get()
        elif self._get_object(obs, 3, 3) == 'door': # we are in a normal room
            for x, y in [(2, 4), (4, 4), (2, 5), (4, 5), (2, 6), (4, 6)]:
                object = self._get_object(obs, x, y)
                color = self._get_color(obs, x, y)
                if object != 'empty': self.objects_seen[(object, color)].append(self.timestep)
                assert object != 'door' and object != 'unseen' and object != 'wall'
            action = Actions.forward
            future_action_list = [Actions.forward, Actions.toggle, Actions.forward]
            for a in future_action_list:
                self.action_queue.put(a)
        elif self._get_object(obs, 2, 3) == 'door': # we are in a test room
            object, color = self._get_object(obs, 3, 4), self._get_color(obs, 3, 4)
            for past_timestep in self.objects_seen[(object, color)]:
                self.memory_associations.append((2 * self.timestep, 2 * past_timestep)) # multiply by 2 because observations are at even indicies and actions are at odd indicies
            if len(self.objects_seen[(object, color)]) % 2 == 0:
                action = Actions.left
                future_action_list = [Actions.forward, Actions.right, Actions.forward, Actions.forward, Actions.toggle, Actions.forward]
                for a in future_action_list:
                    self.action_queue.put(a)
            else:
                action = Actions.right
                future_action_list = [Actions.forward, Actions.left, Actions.forward, Actions.forward, Actions.toggle, Actions.forward]
                for a in future_action_list:
                    self.action_queue.put(a)
        else:
            pass

        assert action is not None

        self.timestep += 1
        return action
    
    def get_memory_associations(self):
        return self.memory_associations