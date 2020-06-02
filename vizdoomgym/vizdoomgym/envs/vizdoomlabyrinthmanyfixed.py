from vizdoomgym.envs.vizdoomenv import VizdoomEnv


class VizdoomLabyrinthManyFixed(VizdoomEnv):

    def __init__(self):
        super(VizdoomLabyrinthManyFixed, self).__init__(13)

class VizdoomLabyrinthManyFixedAngle(VizdoomEnv):

    def __init__(self):
        super(VizdoomLabyrinthManyFixedAngle, self).__init__(14)
