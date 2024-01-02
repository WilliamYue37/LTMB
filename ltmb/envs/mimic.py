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
    def __init__(self, length=5, screen_size=640, **kwargs):
        self.length = length # number of commands
        self.object = self._rand_elem([Ball, Key, Box])
        self.color = self._rand_elem(COLOR_NAMES)
        self.color_to_action = {}
        self.action_to_color = {}

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=7,
            height=7,
            see_through_walls=True,
            max_steps=length,
            screen_size=screen_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return 'Perform the same action for each key color. Do not perform the same action for any other color object.'

    def _gen_grid(self, width, height):
        self.mission = 'Perform the same action for each key color. Do not perform the same action for any other color object.'
        self.grid = Grid(width, height)

        # Fix the player's start position and orientation
        self.agent_pos = np.array((3, 6))
        self.agent_dir = 3 # facing up

        # Place target object in the start room
        self.grid.set(3, 3, self.object(self.color))

    def reset(self, **kwargs):
        self.color = self._rand_elem(COLOR_NAMES)
        self.object = self._rand_elem([Ball, Key, Box])
        self.grid.set(3, 3, self.object(self.color))
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

        # generate a new random color and object
        if valid_move:
            self.color = self._rand_elem(COLOR_NAMES)
            self.object = self._rand_elem([Ball, Key, Box])
            self.grid.set(3, 3, self.object(self.color))
        else:
            self.grid.set(3, 3, None)

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
    
def main():
    env = MimicEnv(length=10, screen_size=800, render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()


