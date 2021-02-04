import random
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp


class Policy:

    def __init__(self):

        pass

    def sample(self):

        return


class EGreedy(Policy):

    def __init__(self, epsilon):
        super().__init__()

        self.epsilon = epsilon

    def sample(self, actions, legal_actions):
        """

        :param q_function:
        :param state:
        :param action_space:
        :return:
        """

        greedy = random.random() > self.epsilon

        optimal_probs = (tf.exp(actions) / tf.reduce_sum(tf.exp(actions))) * tf.cast(
            tf.reshape((legal_actions - 1) * -1, (9, 1)), dtype=tf.float32)

        if greedy:
            idx = tf.argmax(optimal_probs, axis=0)
        else:
            print('Just did a rando')

            s = tfp.distributions.Normal(1, 1)
            samples = tf.abs(s.sample((9,1)))

            optimal_probs *= samples

            idx = tf.argmax(optimal_probs, axis=0)

        action = tf.one_hot(idx, tf.shape(actions)[0])

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

        action = tf.one_hot(tf.argmax(actions), tf.shape(actions)[0])



        '''
        optimal_probs = (tf.exp(actions) / tf.reduce_sum(tf.exp(actions))) * tf.cast(
            tf.reshape((legal_actions - 1) * -1, (9, 1)), dtype=tf.float32)

        action = tf.one_hot(tf.argmax(optimal_probs, axis=0), 9)'''

        return action


class Greedy2(Policy):

    def __init__(self):
        super().__init__()

    def sample(self, actions, legal_actions):
        """

        :param q_function:
        :param state:
        :param action_space:
        :return:
        """

        optimal_probs = (tf.exp(actions) / tf.reduce_sum(tf.exp(actions))) * tf.cast(
            tf.reshape((legal_actions - 1) * -1, (9, 1)), dtype=tf.float32)

        action = tf.one_hot(tf.argmax(optimal_probs, axis=0), 9)

        return action

