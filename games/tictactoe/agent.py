import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

from agents.replay_buffer import ReplayBuffer
from agents.memory_buffer import MemoryBuffer

BATCH_SIZE = 32


class AI:

    def __init__(self, agent_id, behavior, replay_buffer=True):

        self.id = agent_id
        self.behavior = behavior
        self.replay_buffer = ReplayBuffer() if replay_buffer else None
        self.memory_buffer = MemoryBuffer(['state', 'action', 'action_space'], 2)

        self.turn = 0
        self.episode = 0

        self._turn_rewards = None

    def play(self, state, mask, explore=True):
        """

        :param state:
        :param mask:
        :param explore:
        :return:
        """

        self.turn += 1

        state = tf.convert_to_tensor(state)
        mask = tf.convert_to_tensor(mask)

        optimal_action = self.behavior.predict(state, mask, explore)

        optimal_action = optimal_action.numpy().reshape(-1)
        self.memory_buffer.store(action=optimal_action)

        return optimal_action

    def observe(self, state, reward, action_space, done):

        if self.memory_buffer.size > 0:

            old_state, old_action, old_action_space = self.memory_buffer.retrieve(1)

            self.replay_buffer.add_sample(state=old_state,
                                          action=old_action,
                                          action_space=old_action_space,
                                          reward=reward,
                                          next_state=state,
                                          next_action_space=action_space,
                                          done=done)
            self._turn_rewards.append(reward)

        self.memory_buffer.store(state=state.copy().reshape(-1), action_space=action_space.copy())

        return

    def reset(self):

        if self.turn is not None:
            self._turn_rewards = []
        self.turn = 0
        self.memory_buffer.reset()

        return

    def save_model(self):

        self.behavior.save_model(self.id)

        return

    def load_model(self):

        self.behavior.load_model()

        return

    def update(self):

        if self.replay_buffer.current_size != 0:
            data = self.replay_buffer.sample(size=BATCH_SIZE)
            loss = tf.reduce_mean(self.behavior.train(*data))
        else:
            loss = None
            print('Didn\'t update because no samples available.')

        self.episode += 1

        return loss

