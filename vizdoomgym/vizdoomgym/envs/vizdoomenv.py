import os
import gym
from gym import spaces
from vizdoom import *
import numpy as np

# Format (config, scenario, num_actions, difficulty)
CONFIGS = [['basic.cfg', 'basic.wad', 3, 5],                                       # 0
           ['deadly_corridor.cfg', 'deadly_corridor.wad', 6, 1],                   # 1
           ['defend_the_center.cfg', 'defend_the_center.wad', 3, 5],               # 2
           ['defend_the_line.cfg', 'defend_the_line.wad', 3, 5],                   # 3
           ['health_gathering.cfg', 'health_gathering.wad', 3, 5],                 # 4
           ['my_way_home.cfg', 'my_way_home_dense.wad', 3, 5],                     # 5
           ['predict_position.cfg', 'predict_position.wad', 3, 3],                 # 6
           ['take_cover.cfg', 'take_cover.wad', 2, 5],                             # 7
           ['deathmatch.cfg', 'deathmatch.wad', 42, 5],                            # 8
           ['my_way_home.cfg', 'my_way_home_sparse.wad', 3, 5],                    # 9
           ['my_way_home.cfg', 'my_way_home_verysparse.wad', 3, 5],                # 10
           ['health_gathering_supreme.cfg', 'health_gathering_supreme.wad', 3, 5], # 11
           ['labyrinth.cfg', 'labyrinth_single.wad', 3, 5],                        # 12
           ['labyrinth.cfg', 'labyrinth_many_fixed.wad', 3, 5],                    # 13
           ['labyrinth.cfg', 'labyrinth_many.wad', 3, 5],                          # 14
           ['labyrinth.cfg', 'labyrinth_randtx.wad', 3, 5],                        # 15
           ['labyrinth1.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 16
           ['labyrinth2.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 17
           ['labyrinth3.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 18
           ['labyrinth4.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 19
           ['labyrinth5.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 20
           ['labyrinth6.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 21
           ['labyrinth7.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 22
           ['labyrinth8.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 23
           ['labyrinth9.cfg', 'labyrinth_randtx2.wad', 3, 5],                      # 24
           ['labyrinth10.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 25
           ['labyrinth11.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 26
           ['labyrinth12.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 27
           ['labyrinth13.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 28
           ['labyrinth14.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 29
           ['labyrinth15.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 30
           ['labyrinth16.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 31
           ['labyrinth17.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 32
           ['labyrinth18.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 33
           ['labyrinth19.cfg', 'labyrinth_randtx2.wad', 3, 5],                     # 34
           ['labyrinth20.cfg', 'labyrinth_randtx2.wad', 3, 5]]                     # 35


# Maps actions from deprecated ppaquette package to new vizdoom environment
my_way_home_map = {13:[1, 0, 0, 0], 14:[0, 1, 0, 0], 15:[0, 0, 1, 0], 'noop':[0, 0, 0, 1]} 

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
        self.buffer_dims = None
        self.level = level

        self.action_space = spaces.MultiDiscrete([[0, 1]] * 38 + [[-10, 10]] * 2 + [[-100, 100]] * 3)
        self.allowed_actions = list(range(NUM_ACTIONS))
        self.num_actions = CONFIGS[level][2]
        self.observation_space = spaces.Box(0, 255, (self.game.get_screen_height(),
                                                     self.game.get_screen_width(),
                                                     self.game.get_screen_channels()))
        self.viewer = None

    def step(self, action):
        # convert action to vizdoom action space (length 43 one hot array)
        if len(self.allowed_actions) > 0:
            list_action = [int(action[action_idx]) for action_idx in self.allowed_actions]
        else:
            list_action = [int(x) for x in action] 

        if 1 in list_action: chosen_action = my_way_home_map[list_action.index(1)]
        else: chosen_action = my_way_home_map['noop']

        reward = self.game.make_action(chosen_action)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        if state == None: return np.transpose(np.zeros(self.buffer_dims), (1, 2, 0)), reward, True, {'dummy' : 0}
        if not done: observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else: observation = np.uint8(np.zeros(self.observation_space.shape))

        return observation, reward, done, {'dummy' : 0}

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        self.buffer_dims = np.shape(self.state.screen_buffer)
        img = self.state.screen_buffer
        return np.transpose(img, (1, 2, 0))

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        try:
            if self.game.get_state() == None: img = np.zeros(self.buffer_dims, dtype=np.uint8)
            else: img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])
            if mode == 'rgb_array':
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
