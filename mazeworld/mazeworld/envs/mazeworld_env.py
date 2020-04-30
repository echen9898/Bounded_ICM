"""An example implementation of pycolab games as environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym import spaces

from pycolab.examples import better_scrolly_maze

from mazeworld.envs import pycolab_env

class MazeWorld(pycolab_env.PyColabEnv):
    """Custom maze world game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=10,
                 default_reward=-1.):
        self.level = level
        super(MazeWorld, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=8)

    def make_game(self):
        return better_scrolly_maze.make_game(self.level)
