'''
Script to test if mario installation works fine. It
displays the game play simultaneously.
'''

from __future__ import print_function
import gym, universe
import env_wrapper
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario import wrappers
import numpy as np
import time
# from pyvirtualdisplay import Display
from PIL import Image
import utils
import pdb

print('GYM VERSION: {}'.format(gym.__version__))

# virtual display (headless remotes)
# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()

outputdir = './mario_test/'
env_id = 'ppaquette/SuperMarioBros-1-1-v0'
env = gym.make(env_id)
modewrapper = wrappers.SetPlayingMode('human')
acwrapper = wrappers.ToDiscrete()
env = modewrapper(acwrapper(env))
env = env_wrapper.MarioEnv(env)
env = gym.wrappers.Monitor(env, outputdir, video_callable=lambda episode_id: True, force=True)

freshape = fshape = (42, 42)
env.seed(None)
env = env_wrapper.NoNegativeRewardEnv(env)
env = env_wrapper.BufferedObsEnv(env, n=4, skip=1, shape=fshape, channel_last=True)

start = time.time()
episodes = 0
maxepisodes = 1
env.reset()
print('Starting episode 1')
while(1):
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        episodes += 1
        print('Ep: %d, Distance: %d'%(episodes, info['distance']))
        if episodes >= maxepisodes:
            break
        env.reset()
end = time.time()
print('\nTotal Time spent: %0.2f seconds'% (end-start))
print('Done!')