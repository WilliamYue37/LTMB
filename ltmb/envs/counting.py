from __future__ import annotations

import numpy as np
from collections import defaultdict

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key, Box, Door
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

class CountingEnv(MiniGridEnv):
    def __init__(self, length=5, test_freq = 0.3, empty_freq = 0.1, screen_size=640, **kwargs):
        if length < 1:
            raise ValueError('length must be greater than 0')
        if test_freq < 0 or test_freq > 1:
            raise ValueError('test_freq must be between 0 and 1')
        if empty_freq < 0 or empty_freq > 1:
            raise ValueError('empty_freq must be between 0 and 1')

        self.length = length # number of rooms
        max_steps = 7 * length
        self.object_count = defaultdict(int) # count of objects
        self.test_freq = test_freq # frequency of test rooms
        self.empty_freq = empty_freq # frequency of empty object cells
        self.rooms_visited = 1 # number of rooms visited

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=5,
            height=5,
            see_through_walls=True,
            max_steps=max_steps,
            screen_size=screen_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return 'Go through the green door if there are an even number of objects of the same color as the object in the room, otherwise go through the red door'
    
    def _gen_normal_room(self):
        # Fix the player's start position and orientation
        self.agent_pos = np.array((2, 3))
        self.agent_dir = 3 # facing up

        # Place door
        self.grid.set(2, 0, Door('blue'))

        # generate objects
        for x, y in [(1, 1), (3, 1), (1, 2), (3, 2), (1, 3), (3, 3)]:
            object = self._rand_elem([Ball, Key, Box])
            color = self._rand_elem(COLOR_NAMES)
            if self._rand_float(0, 1) > self.empty_freq: 
                self.grid.set(x, y, object(color))
                self.object_count[(object, color)] += 1
    
    def _gen_test_room(self):
        # Fix the player's start position and orientation
        self.agent_pos = np.array((2, 3))
        self.agent_dir = 3 # facing up

        # Place doors
        self.grid.set(1, 0, Door('green'))
        self.grid.set(3, 0, Door('red'))

        # generate object
        object = self._rand_elem([Ball, Key, Box])
        color = self._rand_elem(COLOR_NAMES)
        self.grid.set(2, 1, object(color))
        self.correct_door = (1, 0) if self.object_count[(object, color)] % 2 == 0 else (3, 0)

    def _clear_room(self):
        # clear room
        for i in range(1, 4):
            for j in range(1, 4):
                self.grid.set(i, j, None)
        # clear doors
        self.grid.horz_wall(0, 0)

    def _gen_grid(self, width, height):
        self.mission = 'Go through the green door if there are an even number of objects of the same color as the object in the room, otherwise go through the red door'
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self._gen_normal_room()

    def reset(self, **kwargs):
        self.object_count = defaultdict(int)
        self.rooms_visited = 1
        return super().reset(**kwargs)

    def step(self, action):
        if action == Actions.pickup: # Don't allow picking up objects
            action = Actions.toggle

        obs, reward, terminated, truncated, info = super().step(action)

        if self.agent_pos[1] == 0: # reached a door
            # verify the results of a test room
            if self.agent_pos != (2, 0): # previous room was a test room
                if self.correct_door != self.agent_pos: # wrong door
                    reward = 0
                    info['success'] = False
                    terminated = True
                    return obs, reward, terminated, truncated, info
                
            if self.rooms_visited == self.length: # visited all the rooms
                terminated = True
                reward = 1
                info['success'] = True
                return obs, reward, terminated, truncated, info

            # Generate a new room
            self._clear_room()
            if self._rand_float(0, 1) <= self.test_freq: # generate test room
                self._gen_test_room()
            else: # generate normal room
                self._gen_normal_room()

            self.rooms_visited += 1
        
        if truncated or terminated:
            reward = 1
            info['success'] = True
            
        return self.gen_obs(), reward, terminated, truncated, info
    
def main():
    env = CountingEnv(length=10, screen_size=800, render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()


