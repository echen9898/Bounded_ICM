from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.envs.registration import register

from mazeworld.envs import MazeWorld


register(
    id='Maze-1-v0',
    entry_point='mazeworld.envs:MazeWorld',
    max_episode_steps=1000,
    kwargs={'level': 0, 'max_iterations': 1000})