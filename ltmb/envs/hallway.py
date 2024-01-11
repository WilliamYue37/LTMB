from __future__ import annotations

import numpy as np

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key, Wall, Door, Box
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

class HallwayEnv(MiniGridEnv):
    def __init__(self, length=5, max_steps=16, tile_size=12, screen_size=640, **kwargs):
        self.length = length # number of vertical hallways
        self.size = 4 * length + 5
        max_steps = max(max_steps, self.size + 20)

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            see_through_walls=True,
            max_steps=max_steps,
            screen_size=screen_size,
            tile_size=tile_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return 'Enter the hallway with the same object as the one in the start room'
    
    def _rand_obj(self):
        """Randomly generate an object that is not the target object and color"""
        obj = self._rand_elem([Ball, Key, Box])
        color = self._rand_elem(COLOR_NAMES)
        # If the object and color are the same as the target, change the color
        if obj == self.target_obj and color == self.target_color: 
            new_colors = [c for c in COLOR_NAMES if c != self.target_color]
            color = self._rand_elem(new_colors)
        return obj(color)

    def _gen_grid(self, width, height):
        self.mission = 'Enter the hallway with the same object as the one in the start room'
        self.grid = Grid(width, height)

        # choose the target object and color
        self.target_color = self._rand_elem(COLOR_NAMES)
        self.target_obj = self._rand_elem([Key, Ball, Box])

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2

        # Start room
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(5, width - 2):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallways
        for i in range(6, width - 2, 4):
            # set walls
            for j in range(2):
                self.grid.set(i, upper_room_wall - j, Wall()) # upper left wall
                self.grid.set(i + 2, upper_room_wall - j, Wall()) # upper right wall
                self.grid.set(i, lower_room_wall + j, Wall()) # lower left wall
                self.grid.set(i + 2, lower_room_wall + j, Wall()) # lower right wall
            self.grid.set(i + 1, upper_room_wall - 1, Wall())
            self.grid.set(i + 1, lower_room_wall + 1, Wall())
            
            # set doors
            self.grid.set(i + 1, upper_room_wall + 1, Door(self._rand_elem(COLOR_NAMES)))
            self.grid.set(i + 1, lower_room_wall - 1, Door(self._rand_elem(COLOR_NAMES)))

            # set objects
            self.grid.set(i + 1, upper_room_wall, self._rand_obj())
            self.grid.set(i + 1, lower_room_wall, self._rand_obj())

        # Fix the player's start position and orientation
        self.agent_pos = np.array((2, height // 2))
        self.agent_dir = 0

        # Place target object in the start room
        self.grid.set(1, height // 2 - 1, self.target_obj(self.target_color))

        # Choose the target hallway and place the target object there
        self.target_hallway = self._rand_int(0, self.length)
        if self._rand_int(0, 2) == 0: # Place the target object in the upper hallway
            self.target_pos = (7 + 4 * self.target_hallway, upper_room_wall)
            self.success_pos = (7 + 4 * self.target_hallway , upper_room_wall + 1)
        else: # Place the target object in the lower hallway
            self.target_pos = (7 + 4 * self.target_hallway, lower_room_wall)
            self.success_pos = (7 + 4 * self.target_hallway , lower_room_wall - 1)
        self.grid.set(*self.target_pos, self.target_obj(self.target_color))

    def step(self, action):
        if action == Actions.pickup: # Don't allow picking up objects
            action = Actions.toggle
        obs, reward, terminated, truncated, info = super().step(action)

        # check if agent has entered a hallway
        if isinstance(self.grid.get(*self.agent_pos), Door):
            if tuple(self.agent_pos) == self.success_pos:
                reward = self._reward()
                info['success'] = True
            else:
                reward = 0
                info['success'] = False
            terminated = True
     
        return obs, reward, terminated, truncated, info
    
    def get_obs_render(self):
        return self.get_pov_render(tile_size=self.tile_size)
    
def main():
    env = HallwayEnv(length=5, screen_size=800, render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()