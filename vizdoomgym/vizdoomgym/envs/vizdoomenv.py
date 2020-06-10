import os
from time import sleep
import multiprocessing
import logging

import numpy as np

import gym
from gym import spaces, error
from gym.utils import seeding

from vizdoom import *

logger = logging.getLogger(__name__)

# Arguments:
RANDOMIZE_MAPS = 80  # 0 means load default, otherwise randomly load in the id mentioned
NO_MONSTERS = True  # remove monster spawning

# Constants
NUM_ACTIONS = 43
NUM_LEVELS = 9
CONFIG = 0
SCENARIO = 1
MAP = 2
DIFFICULTY = 3
ACTIONS = 4
MIN_SCORE = 5
TARGET_SCORE = 6

# Format (config, scenario, map, difficulty, actions, min, target)
DOOM_SETTINGS = [
    ['basic.cfg', 'basic.wad', 'map01', 5, [0, 10, 11], -485, 10],                                   # 0 - Basic
    ['deadly_corridor.cfg', 'deadly_corridor.wad', '', 1, [0, 10, 11, 13, 14, 15], -120, 1000],      # 1 - Corridor
    ['defend_the_center.cfg', 'defend_the_center.wad', '', 5, [0, 14, 15], -1, 10],                  # 2 - DefendCenter
    ['defend_the_line.cfg', 'defend_the_line.wad', '', 5, [0, 14, 15], -1, 15],                      # 3 - DefendLine
    ['health_gathering.cfg', 'health_gathering.wad', 'map01', 5, [13, 14, 15], 0, 1000],             # 4 - HealthGathering
    ['my_way_home.cfg', 'my_way_home_dense.wad', '', 5, [13, 14, 15], -0.22, 0.5],                   # 5 - MyWayHome
    ['predict_position.cfg', 'predict_position.wad', 'map01', 3, [0, 14, 15], -0.075, 0.5],          # 6 - PredictPosition
    ['take_cover.cfg', 'take_cover.wad', 'map01', 5, [10, 11], 0, 750],                              # 7 - TakeCover
    ['deathmatch.cfg', 'deathmatch.wad', '', 5, [x for x in range(NUM_ACTIONS) if x != 33], 0, 20],  # 8 - Deathmatch
    ['my_way_home.cfg', 'my_way_home_sparse.wad', '', 5, [13, 14, 15], -0.22, 0.5],                  # 9 - MyWayHomeFixed
    ['my_way_home.cfg', 'my_way_home_verysparse.wad', '', 5, [13, 14, 15], -0.22, 0.5],              # 10 - MyWayHomeFixed15
    ['labyrinth.cfg', 'labyrinth_single.wad', 'map01', 5, [13, 14, 15], -0.22, 0.5],                 # 11 - Labyrinth Map - one map
    ['labyrinth.cfg', 'labyrinth_many.wad', 'map01', 5, [13, 14, 15], -0.22, 0.5],                   # 12 - Labyrinth Map - many maps, random spawn, random angle
    ['labyrinth.cfg', 'labyrinth_many_fixed.wad', 'map01', 5, [13, 14, 15], -0.22, 0.5],             # 13 - Labyrinth Map - many maps, fixed spawn, random angle
    ['labyrinth.cfg', 'labyrinth_many_fixed2.wad', 'map01', 5, [13, 14, 15], -0.22, 0.5]             # 14 - Labyrinth Map - many maps, fixed spawn, fixed angle
]

# Singleton pattern
class DoomLock:
    class __DoomLock:
        def __init__(self):
            self.lock = multiprocessing.Lock()
    instance = None
    def __init__(self):
        if not DoomLock.instance:
            DoomLock.instance = DoomLock.__DoomLock()
    def get_lock(self):
        return DoomLock.instance.lock


class VizdoomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self, level):
        self.previous_level = -1
        self.level = level
        self.game = DoomGame()
        self.doom_dir = os.path.dirname(os.path.abspath(__file__))
        self.scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self._mode = 'algo'                         # 'algo' or 'human'
        self.no_render = False                      # To disable double rendering in human mode
        self.viewer = None
        self.is_initialized = False                 # Indicates that reset() has been called
        self.curr_seed = 0
        self.lock = (DoomLock()).get_lock()
        self.action_space = spaces.MultiDiscrete([[0, 1]] * 38 + [[-10, 10]] * 2 + [[-100, 100]] * 3)
        self.allowed_actions = list(range(NUM_ACTIONS))
        self.screen_height = 480
        self.screen_width = 640
        self.screen_resolution = ScreenResolution.RES_640X480
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self._seed()
        self._configure()

    def _configure(self, lock=None, **kwargs):
        if 'screen_resolution' in kwargs:
            logger.warn('Deprecated - Screen resolution must now be set using a wrapper. See documentation for details.')
        # Multiprocessing lock
        if lock is not None:
            self.lock = lock

    def _load_level(self):
        # Closing if is_initialized
        if self.is_initialized:
            self.is_initialized = False
            self.game.close()
            self.game = DoomGame()

        # Customizing level
        if getattr(self, '_customize_game', None) is not None and callable(self._customize_game):
            self.level = -1
            self._customize_game()

        else:
            # Common settings
            self.game.load_config(os.path.join(self.doom_dir, 'scenarios/%s' % DOOM_SETTINGS[self.level][CONFIG]))
            self.game.set_doom_scenario_path(os.path.join(self.scenarios_dir, DOOM_SETTINGS[self.level][SCENARIO]))
            if DOOM_SETTINGS[self.level][MAP] != '':
                if RANDOMIZE_MAPS > 0 and 'labyrinth' in DOOM_SETTINGS[self.level][CONFIG].lower():
                    if 'fix' in DOOM_SETTINGS[self.level][SCENARIO].lower():
                        # mapId = 'map%02d'%np.random.randint(1, 23)
                        mapId = 'map%02d'%np.random.randint(4, 8)
                    else:
                        mapId = 'map%02d'%np.random.randint(1, RANDOMIZE_MAPS+1)
                    print('\t=> Special Config: Randomly Loading Maps. MapID = ' + mapId)
                    self.game.set_doom_map(mapId)
                else:
                    print('\t=> Default map loaded. MapID = ' + DOOM_SETTINGS[self.level][MAP])
                    self.game.set_doom_map(DOOM_SETTINGS[self.level][MAP])
            self.game.set_doom_skill(DOOM_SETTINGS[self.level][DIFFICULTY])
            self.allowed_actions = DOOM_SETTINGS[self.level][ACTIONS]
            self.game.set_screen_resolution(self.screen_resolution)

        self.previous_level = self.level
        self._closed = False

        # Algo mode
        if 'human' != self._mode:
            if NO_MONSTERS:
                print('\t=> Special Config: Monsters Removed.')
                self.game.add_game_args('-nomonsters 1')
            # self.game
            self.game.set_window_visible(False)
            self.game.set_mode(Mode.PLAYER)
            self.no_render = False
            try:
                with self.lock:
                    self.game.init()
            except (ViZDoomUnexpectedExitException, ViZDoomErrorException):
                raise error.Error(
                    'VizDoom exited unexpectedly. This is likely caused by a missing multiprocessing lock. ' +
                    'To run VizDoom across multiple processes, you need to pass a lock when you configure the env ' +
                    '[e.g. env.configure(lock=my_multiprocessing_lock)], or create and close an env ' +
                    'before starting your processes [e.g. env = gym.make("DoomBasic-v0"); env.close()] to cache a ' +
                    'singleton lock in memory.')
            self._start_episode()
            self.is_initialized = True
            return self.game.get_state().screen_buffer.copy()

        # Human mode
        else:
            if NO_MONSTERS:
                print('\t=> Special Config: Monsters Removed.')
                self.game.add_game_args('-nomonsters 1')
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.SPECTATOR)
            self.no_render = True
            with self.lock:
                self.game.init()
            self._start_episode()
            self.is_initialized = True
            self._play_human_mode()
            return np.zeros(shape=self.observation_space.shape, dtype=np.uint8)

    def _start_episode(self):
        if self.curr_seed > 0:
            self.game.set_seed(self.curr_seed)
            self.curr_seed = 0
        self.game.new_episode()
        return

    def _play_human_mode(self):
        while not self.game.is_episode_finished():
            self.game.advance_action()
            state = self.game.get_state()
            total_reward = self.game.get_total_reward()
            info = self._get_game_variables(state.game_variables)
            info["TOTAL_REWARD"] = round(total_reward, 4)
            print('===============================')
            print('State: #' + str(state.number))
            print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
            print('Reward: \t' + str(self.game.get_last_reward()))
            print('Total Reward: \t' + str(total_reward))
            print('Variables: \n' + str(info))
            sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
        print('===============================')
        print('Done')
        return

    def _step(self, action):
        if NUM_ACTIONS != len(action):
            logger.warn('Doom action list must contain %d items. Padding missing items with 0' % NUM_ACTIONS)
            old_action = action
            action = [0] * NUM_ACTIONS
            for i in range(len(old_action)):
                action[i] = old_action[i]
        # action is a list of numbers but DoomGame.make_action expects a list of ints
        if len(self.allowed_actions) > 0:
            list_action = [int(action[action_idx]) for action_idx in self.allowed_actions]
        else:
            list_action = [int(x) for x in action]
        try:
            reward = self.game.make_action(list_action)
            state = self.game.get_state()

            if state is None: # episode finished
                is_finished = True
                info = {'TOTAL_REWARD' : round(self.game.get_total_reward(), 4)}
                return np.zeros(shape=self.observation_space.shape, dtype=np.uint8), reward, is_finished, info

            info = self._get_game_variables(state.game_variables)
            info["TOTAL_REWARD"] = round(self.game.get_total_reward(), 4)

            if self.game.is_episode_finished():
                is_finished = True
                return np.zeros(shape=self.observation_space.shape, dtype=np.uint8), reward, is_finished, info
            else:
                is_finished = False
                return np.transpose(state.screen_buffer.copy(), (1, 2, 0)), reward, is_finished, info

        except vizdoom.ViZDoomIsNotRunningException:
            return np.zeros(shape=self.observation_space.shape, dtype=np.uint8), 0, True, {}

    def _reset(self):
        if self.is_initialized and not self._closed:
            self._start_episode()
            screen_buffer = self.game.get_state().screen_buffer
            if screen_buffer is None:
                raise error.Error(
                    'VizDoom incorrectly initiated. This is likely caused by a missing multiprocessing lock. ' +
                    'To run VizDoom across multiple processes, you need to pass a lock when you configure the env ' +
                    '[e.g. env.configure(lock=my_multiprocessing_lock)], or create and close an env ' +
                    'before starting your processes [e.g. env = gym.make("DoomBasic-v0"); env.close()] to cache a ' +
                    'singleton lock in memory.')
            observation = screen_buffer.copy()
        else:
            observation = self._load_level()
        return np.transpose(observation, (1, 2, 0))

    def _render(self, mode='rgb_array', close=False):
        try:
            if 'human' == mode and self.no_render:
                return

            # emulator returns None state when episode is finished
            if self.game.get_state() == None: img = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
            else: img = np.transpose(self.game.get_state().screen_buffer, (1, 2, 0))

            if mode == 'rgb_array':
                return img
            elif mode is 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        except vizdoom.ViZDoomIsNotRunningException:
            pass  # Doom has been closed

    def _close(self):
        # Lock required for VizDoom to close processes properly
        with self.lock:
            self.game.close()

    def _seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed) % 2 ** 32
        return [self.curr_seed]

    def _get_game_variables(self, state_variables):
        info = {
            "LEVEL": self.level
        }
        if state_variables is None:
            return info
        info['KILLCOUNT'] = state_variables[0]
        info['ITEMCOUNT'] = state_variables[1]
        info['SECRETCOUNT'] = state_variables[2]
        info['FRAGCOUNT'] = state_variables[3]
        info['HEALTH'] = state_variables[4]
        info['ARMOR'] = state_variables[5]
        info['DEAD'] = state_variables[6]
        info['ON_GROUND'] = state_variables[7]
        info['ATTACK_READY'] = state_variables[8]
        info['ALTATTACK_READY'] = state_variables[9]
        info['SELECTED_WEAPON'] = state_variables[10]
        info['SELECTED_WEAPON_AMMO'] = state_variables[11]
        info['AMMO1'] = state_variables[12]
        info['AMMO2'] = state_variables[13]
        info['AMMO3'] = state_variables[14]
        info['AMMO4'] = state_variables[15]
        info['AMMO5'] = state_variables[16]
        info['AMMO6'] = state_variables[17]
        info['AMMO7'] = state_variables[18]
        info['AMMO8'] = state_variables[19]
        info['AMMO9'] = state_variables[20]
        info['AMMO0'] = state_variables[21]
        info['POSITION_X'] = doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER1))
        info['POSITION_Y'] = doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER2))
        return info
