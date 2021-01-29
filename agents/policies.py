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

    def sample(self, actions):
        """

        :param q_function:
        :param state:
        :param action_space:
        :return:
        """

        greedy = random.random() > self.epsilon

        if greedy:
            idx = np.argmax(actions)
        else:
            idx = random.randrange(actions.shape[0])
            print(idx)

        action = np.zeros((1, actions.shape[0]))
        action[:, idx] = 1

        return action


class Greedy(Policy):

    def __init__(self):
        super().__init__()

    def sample(self, actions):
        """

        :param q_function:
        :param state:
        :param action_space:
        :return:
        """

        idx = np.argmax(actions)

        action = np.zeros((1, actions.shape[0]))
        action[:, idx] = 1

        return action
