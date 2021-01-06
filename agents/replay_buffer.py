import numpy as np

STATE_SIZE =

class ReplayBuffer:

    def __init__(self):

        self.data = np.empty(shape=[])


    def sample(self, size):

        self.