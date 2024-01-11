from __future__ import annotations

import numpy as np

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key, Box
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

class MimicEnv(MiniGridEnv):
    def __init__(self, length=5, empty_freq=0.1, tile_size=12, screen_size=640, **kwargs):
        self.length = length # number of commands
        self.empty_freq = empty_freq # frequency of empty object cells
        self.object = None
        self.color = None
        self.color_to_action = {}
        self.action_to_color = {}
        self.tile_size = tile_size # size of tiles in pixels

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=7,
            height=7,
            see_through_walls=True,
            max_steps=length,
            screen_size=screen_size,
            tile_size=tile_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return 'Perform the same action for each key color. Do not perform the same action for any other color object.'
    
    def _gen_rand_object(self, x: int, y: int):
        """Generate a random object in a random color at position (x, y) and returns the object and color."""
        if self._rand_float(0, 1) > self.empty_freq:
            obj = self._rand_elem([Ball, Key, Box])
            color = self._rand_elem(COLOR_NAMES)
            self.grid.set(x, y, obj(color))
            return obj, color
        else:
            self.grid.set(x, y, None)
            return None, None
        
    def _gen_new_room(self):
        for i in range(7):
            for j in range(7):
                if i == 3 and j == 6: # do not place object in agent's starting position
                    continue
                obj, color = self._gen_rand_object(i, j)
                if i == 3 and j == 3: # save target object
                    self.object = obj
                    self.color = color

    def _gen_grid(self, width, height):
        self.mission = 'Perform the same action for each key color. Do not perform the same action for any other color object.'
        self.grid = Grid(width, height)

        # Fix the player's start position and orientation
        self.agent_pos = np.array((3, 6))
        self.agent_dir = 3 # facing up

        self._gen_new_room()

    def reset(self, **kwargs):
        self._gen_new_room()
        self.color_to_action = {}
        self.action_to_color = {}
        return super().reset(**kwargs)

    def step(self, action):
        valid_move = True
        # check if action is already assigned
        if self.object != Key and action in self.action_to_color:
            # object must be a key of the same color
            if self.object == Key and self.color == self.action_to_color[action]:
                valid_move = True
            else:
                valid_move = False
        elif self.object == Key: # check if color is already assigned if object is a key, otherwise assign color to action
            if self.color in self.color_to_action:
                # action must be the same as the one assigned to the color
                if action == self.color_to_action[self.color]:
                    valid_move = True
                else:
                    valid_move = False
            else:
                self.color_to_action[self.color] = action
                self.action_to_color[action] = self.color
                valid_move = True

        # generate a new room
        self._gen_new_room()

        # Don't allow moving or picking up objects
        action = Actions.drop
        obs, reward, terminated, truncated, info = super().step(action)
        if not valid_move:
            reward = -1
            terminated = True
            info['success'] = False
        elif valid_move and truncated:
            reward = 1
            truncated = False
            terminated = True
            info['success'] = True
     
        return obs, reward, terminated, truncated, info
    
    def get_obs_render(self):
        return self.get_pov_render(tile_size=self.tile_size)
    
def main():
    env = MimicEnv(length=10, screen_size=800, render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()


