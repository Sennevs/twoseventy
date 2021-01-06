import random
import numpy as np


class Policy:

    def __init__(self):

        pass

    def sample(self):

        return


class EGreedy(Policy):

    def __init__(self, epsilon):
        super().__init__()

        self.epsilon = epsilon

    def sample(self, q_function, state, action_space):
        """

        :param q_function:
        :param state:
        :param action_space:
        :return:
        """

        greedy = random.random() > self.epsilon
        as_size = action_space.shape[0]

        state = state.repeat(state, as_size, axis=0)

        preds = q_function([state, action_space])

        idx = np.argmax(preds) if greedy else np.random.choice(as_size, 1)

        action = np.zeros((as_size, 1))
        action[idx] = 1

        return action


class Greedy(Policy):

    def __init__(self):
        super().__init__()

    def sample(self, q_function, state, action_space):
        """

        :param q_function:
        :param state:
        :param action_space:
        :return:
        """

        as_size = action_space.shape[0]

        state = state.repeat(state, as_size, axis=0)

        preds = q_function([state, action_space])

        idx = np.argmax(preds)

        action = np.zeros((as_size, 1))
        action[idx] = 1

        return action
