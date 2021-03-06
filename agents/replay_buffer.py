from collections import deque
import random
import tensorflow as tf
import numpy as np


class ReplayBuffer:

    def __init__(self, max_size=1000):

        self.data = deque()
        self.max_size = max_size
        self.current_size = 0

        return

    def sample(self, size):

        samples = [random.randrange(self.current_size) for _ in range(size)]
        return_data = [self.data[i] for i in samples]
        return_data = list(map(list, zip(*return_data)))
        return_data = [tf.convert_to_tensor(np.vstack(i)) for i in return_data]

        return tuple(return_data)

    def add_sample(self, state, action, reward, next_state, action_space, next_action_space, done):

        self.current_size = min(self.current_size + 1, self.max_size - 1)

        if self.current_size >= self.max_size:
            self.data.popleft()

        self.data.append([state, action, reward, next_state, action_space, next_action_space, done])

        return
