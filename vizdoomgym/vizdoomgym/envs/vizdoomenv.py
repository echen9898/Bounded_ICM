import os
import gym
from gym import spaces
from vizdoom import *
import numpy as np

# Format (config, scenario, num_actions, difficulty)
CONFIGS = [['basic.cfg', 'basic.wad', 3, 5],                            # 0
           ['deadly_corridor.cfg', 'deadly_corridor.wad', 6, 1],        # 1
           ['defend_the_center.cfg', 'defend_the_center.wad', 3, 5],    # 2
           ['defend_the_line.cfg', 'defend_the_line.wad', 3, 5],        # 3
           ['health_gathering.cfg', 'health_gathering.wad', 3, 5],      # 4
           ['my_way_home.cfg', 'my_way_home_dense.wad', 3, 5],          # 5
           ['predict_position.cfg', 'predict_position.wad', 3, 3],      # 6
           ['take_cover.cfg', 'take_cover.wad', 2, 5],                  # 7
           ['deathmatch.cfg', 'deathmatch.wad', 42, 5],                 # 8
           ['my_way_home.cfg', 'my_way_home_sparse.wad', 3, 5],         # 9
           ['my_way_home.cfg', 'my_way_home_verysparse.wad', 3, 5],     # 10
           ['health_gathering_supreme.cfg', 3, 5]]                      # 11

NUM_ACTIONS = 43


class VizdoomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self, level):

        # init game
        self.game = DoomGame()
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[level][0]))
        self.game.set_doom_scenario_path(os.path.join(scenarios_dir, CONFIGS[level][1]))
        self.game.set_doom_skill(CONFIGS[level][3])
        self.game.set_window_visible(False)
        self.game.init()
        self.state = None
        self.level = level

        self.action_space = spaces.MultiDiscrete([[0, 1]] * 38 + [[-10, 10]] * 2 + [[-100, 100]] * 3)
        self.allowed_actions = list(range(NUM_ACTIONS))
        self.num_actions = CONFIGS[level][2]
        self.observation_space = spaces.Box(0, 255, (self.game.get_screen_height(),
                                                     self.game.get_screen_width(),
                                                     self.game.get_screen_channels()))
        self.viewer = None

    def step(self, action):
        # convert action to vizdoom action space (one hot)
        if len(self.allowed_actions) > 0:
            list_action = [int(action[action_idx]) for action_idx in self.allowed_actions]
        else:
            list_action = [int(x) for x in action] 

        # list action is a 43 length array, one hot encoded at the corresponding action index
        # The following mapping applies to doom my way home only!!
        if 1 in list_action:
            action = list_action.index(1)
            if action == 13:
                chosen_action = [1, 0, 0, 0]
            elif action == 14:
                chosen_action = [0, 1, 0, 0]
            elif action == 15:
                chosen_action = [0, 0, 1, 0]
        else:
            chosen_action = [0, 0, 0, 1]

        reward = self.game.make_action(chosen_action)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        # Generalize this (works only for doom my way home currently)
        if reward > 0.5:
            return np.transpose(state.screen_buffer, (1, 2, 0)), reward, True, {'dummy' : 0}

        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = np.uint8(np.zeros(self.observation_space.shape))

        info = {'dummy': 0}

        return observation, reward, done, info

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer
        return np.transpose(img, (1, 2, 0))

    def render(self, mode='human', close=False):
        if close: # added this clause
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return 
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            if mode == 'rgb_array': # added these clauses
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        except AttributeError:
            pass

    @staticmethod
    def get_keys_to_action():
        # you can press only one key at a time!
        keys = {(): 2,
                (ord('a'),): 0,
                (ord('d'),): 1,
                (ord('w'),): 3,
                (ord('s'),): 4,
                (ord('q'),): 5,
                (ord('e'),): 6}
        return keys
