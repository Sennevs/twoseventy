import numpy as np
import tensorflow as tf
from tensorflow import GradientTape, function
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam, SGD

from agents.q_network import QNetwork
from agents.policies import EGreedy, Greedy
from agents.replay_buffer import ReplayBuffer

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


BATCH_SIZE = 100

class AI:

    def __init__(self, agent_id):

        self.id = agent_id

        # model params - this part needs to be rewritten, so we can dynamically choose the config params
        self.q = QNetwork()
        self.q_target = QNetwork()
        self.discount_factor = 0.99
        self.policy = EGreedy(epsilon=0.01)
        self.policy_greedy = Greedy()
        self.q_optimizer = Adam(0.001)
        self.q_target_optimizer = SGD(0.01)

        self.action_space = [9, 1]
        self.state_space = [9, 1]

        # not sure if this should be list or numpy/tensor
        # this is what we need to keep track of the environment to let our agent reflect and learn from its interactions

        # replay buffer still needs to be implemented and is currently not in use
        self.replay_buffer = ReplayBuffer()
        self.use_rb = False

        self.turn = None

        # metrics
        self.loss_hist = []
        self.reward_hist = []
        self.turn_hist = []

        self._turn_rewards = []

        self.memory = {'state': None, 'action': None, 'action_space': None}

    def play(self, state, legal_actions, greedy=False, target=False):


        self.turn += 1

        legal_idx = (legal_actions == 0)

        actions = np.zeros([9, 9])
        np.fill_diagonal(actions, 1)
        legal_actions = actions[legal_idx]

        state = state.reshape([1, state.shape[0]]).repeat(legal_actions.shape[0], axis=0)

        # loop over legal actions and select action with highest q-value\
        if target:
            optimal_actions = self.q_target([state, legal_actions])
        else:
            optimal_actions = self.q([state, legal_actions])

        if greedy:
            optimal_idx = self.policy_greedy.sample(optimal_actions)
        else:
            optimal_idx = self.policy.sample(optimal_actions)

        optimal_action = np.zeros(9)

        optimal_action[legal_idx] = optimal_idx.reshape(-1)

        self.memory['action'] = optimal_action

        # self.previous_actions.append(optimal_actions)
        return optimal_action

    def observe(self, state, reward, action_space):

        done = action_space is None

        if self.memory['state'] is not None:
            self.replay_buffer.add(**self.memory, reward=reward, next_state=state, next_action_space=self.memory['action_space'], done=done)
            self._turn_rewards.append(reward)

        self.memory['state'] = state
        self.memory['action_space'] = action_space

        return

    def reset(self):

        if self.turn is not None:
            self.turn_hist.append(self.turn)
            self.reward_hist.append(sum(self._turn_rewards))
            self._turn_rewards = []
        self.turn = 0
        self.memory = {'state': None, 'action': None, 'action_space': None}

        return

    def update(self):

        if self.replay_buffer.current_size != 0:
            # should find a way to make the update rule variable
            states, actions, rewards, next_states, action_spaces, next_action_spaces, dones = self.replay_buffer.sample(size=BATCH_SIZE)
            optimal_actions_list = []
            print(next_action_spaces)

            target_q_values = []

            for idx in range(BATCH_SIZE):

                reward = rewards[idx].reshape((1, -1))
                done = dones[idx]
                if done:
                    target_q_value = reward
                else:

                    next_action_space = next_action_spaces[idx].reshape((1, -1))
                    legal_idx = (next_action_space == 0).reshape(-1)

                    next_state = next_states[idx].reshape((1, -1))

                    next_actions_full = np.zeros([9, 9])
                    np.fill_diagonal(next_actions_full, 1)
                    next_legal_actions = next_actions_full[legal_idx]
                    optimal_states = next_state.reshape([1, next_state.shape[1]]).repeat(next_legal_actions.shape[0], axis=0)

                    optimal_actions = self.q_target([optimal_states, next_legal_actions])

                    next_action = self.policy_greedy.sample(optimal_actions)

                    optimal_action = np.zeros((1, 9))

                    optimal_action[:, legal_idx] = next_action

                    optimal_actions_list.append(optimal_action)

                    next_state = next_state.reshape([1, -1])

                    target_q_value = reward + self.discount_factor * self.q([next_state, optimal_action])
                target_q_values.append(target_q_value)

            target_q_values = np.stack(target_q_values)
            # compute td error
            target_q_values = tf.convert_to_tensor(target_q_values, dtype=tf.float32)

            @function(experimental_relax_shapes=True)
            def test_update():

                with GradientTape() as tape:
                    q_values = self.q([states, actions])
                    q_loss = mean_squared_error(target_q_values, q_values)
                grads = tape.gradient(q_loss, self.q.trainable_variables)

                self.q_optimizer.apply_gradients(zip(grads, self.q.trainable_variables))
                # apply soft update
                grads = [(b - a) for a, b in zip(self.q.trainable_variables, self.q_target.trainable_variables)]
                self.q_target_optimizer.apply_gradients(zip(grads, self.q_target.trainable_variables))

                return

        else:
            print('Didn\'t update because no samples available.')

        return

    def save_model(self):

        self.q.save(f'./models/tictactoe/{self.id}/q')
        self.q_target.save(f'./models/tictactoe/{self.id}/q_target')

        return




