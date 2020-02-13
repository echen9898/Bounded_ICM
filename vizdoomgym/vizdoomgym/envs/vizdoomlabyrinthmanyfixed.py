from vizdoomgym.envs.vizdoomenv import VizdoomEnv


class VizdoomLabyrinthManyFixed(VizdoomEnv):

    def __init__(self):
        super(VizdoomLabyrinthManyFixed, self).__init__(13)