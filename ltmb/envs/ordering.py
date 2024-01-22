from __future__ import annotations

import numpy as np
import itertools
import random

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key, Box
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

class OrderingEnv(MiniGridEnv):
    def __init__(self, length=5, tile_size=12, screen_size=640, **kwargs):
        self.length = length # number of commands
        max_steps = 18 + length
        self.tile_size = tile_size # size of tiles in pixels
        random.seed(int(self._rand_int(0, 10**9)))
        self.permutation = list(itertools.product([Ball, Key, Box], COLOR_NAMES))
        self.timestep = 0
        self.choices = []
        

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=7,
            height=7,
            see_through_walls=True,
            max_steps=max_steps,
            screen_size=screen_size,
            tile_size=tile_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return 'Memorize the order of the first 18 colored objects to appear. When shown two objects, select the one that appeared first.'
    
    def _gen_new_room(self):
        if self.timestep < 18:
            object, color = self.permutation[self.timestep]
            self.grid.set(3, 3, object(color))
        else:
            self.grid.set(3, 3, None)
            self.choices = random.sample(self.permutation, 2)
            self.grid.set(2, 3, self.choices[0][0](self.choices[0][1]))
            self.grid.set(4, 3, self.choices[1][0](self.choices[1][1]))

    def _gen_grid(self, width, height):
        self.mission = 'Memorize the order of the first 18 colored objects to appear. When shown two objects, select the one that appeared first.'
        self.grid = Grid(width, height)

        # Fix the player's start position and orientation
        self.agent_pos = np.array((3, 6))
        self.agent_dir = 3 # facing up

        # generate a permutation of all possible objects and colors
        random.shuffle(self.permutation)

        self.timestep = 0
        self._gen_new_room()

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        incorrect_action = False
        if self.timestep >= 18:
            correct_action = Actions.left if self.permutation.index(self.choices[0]) < self.permutation.index(self.choices[1]) else Actions.right
            incorrect_action = action != correct_action

        # generate a new room
        self.timestep += 1
        self._gen_new_room()

        # Don't allow moving or picking up objects
        action = Actions.drop
        obs, reward, terminated, truncated, info = super().step(action)
        if incorrect_action:
            reward = -1
            terminated = True
            info['success'] = False
        elif self.timestep == 18 + self.length:
            reward = 1
            truncated = False
            terminated = True
            info['success'] = True
     
        return obs, reward, terminated, truncated, info
    
    def get_obs_render(self):
        return self.get_pov_render(tile_size=self.tile_size)
    
def main():
    env = OrderingEnv(length=10, tile_size=TILE_PIXELS, screen_size=1300, render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()


