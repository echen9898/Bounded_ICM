from vizdoomgym.envs.vizdoomenv import VizdoomEnv


class VizdoomLabyrinthSingle(VizdoomEnv):

    def __init__(self):
        super(VizdoomLabyrinthSingle, self).__init__(12)