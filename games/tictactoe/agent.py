import numpy as np

from tensorflow import GradientTape
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam, SGD

from agents.q_network import QNetwork
from agents.policies import EGreedy, Greedy
from agents.replay_buffer import ReplayBuffer

BATCH_SIZE = 1

class AI:

    def __init__(self, agent_id):

        self.id = agent_id

        # model params - this part needs to be rewritten, so we can dynamically choose the config params
        self.q = QNetwork()
        self.q_target = QNetwork()
        self.discount_factor = 0.99
        self.policy = EGreedy(epsilon=0.1)
        self.policy_greedy = Greedy()
        self.q_optimizer = Adam(0.001)
        self.q_target_optimizer = SGD(0.001)

        self.action_space = [9, 1]
        self.state_space = [9, 1]

        # not sure if this should be list or numpy/tensor
        # this is what we need to keep track of the environment to let our agent reflect and learn from its interactions

        # replay buffer still needs to be implemented and is currently not in use
        self.replay_buffer = ReplayBuffer()
        self.use_rb = False

        self.loss_hist = []


        self.memory = {'state': None, 'action': None, 'reward': None, 'action_space': None}

    def play(self, state, legal_actions, greedy=False, target=False):

        state = state.reshape([1, 9])

        legal_idx = (legal_actions == 0)[:, 0]

        actions = np.zeros([9, 9])
        np.fill_diagonal(actions, 1)
        legal_actions = actions[legal_idx]

        state = state.repeat(legal_actions.shape[0], axis=0)

        # loop over legal actions and select action with highest q-value\
        if target:
            optimal_actions = self.q_target([state, legal_actions])
        else:
            optimal_actions = self.q([state, legal_actions])

        if greedy:
            optimal_idx = self.policy_greedy.sample(optimal_actions)
        else:
            optimal_idx = self.policy.sample(optimal_actions)

        optimal_action = np.zeros((1, 9))

        optimal_action[:, legal_idx] = optimal_idx

        # self.previous_actions.append(optimal_actions)
        return optimal_action

    def observe(self, state, action, reward, action_space):

        if self.memory['state'] is not None:
            self.replay_buffer.add(**self.memory, next_state=state, next_action_space=action_space)

        self.memory['state'] = state
        self.memory['action'] = action
        self.memory['reward'] = reward
        self.memory['action_space'] = action_space

        return

    def reset(self):

        self.memory = {'state': None, 'action': None, 'reward': None, 'action_space': None}

        return

    def update(self):

        if self.replay_buffer.current_size != 0:
            # should find a way to make the update rule variable
            state, action, reward, next_state, action_space, next_action_space = self.replay_buffer.sample(size=BATCH_SIZE)

            if next_action_space is not None:
                legal_idx = (next_action_space == 0)[:, 0]

                next_actions_full = np.zeros([9, 9])
                np.fill_diagonal(next_actions_full, 1)
                next_legal_actions = next_actions_full[legal_idx]
                optimal_states = next_state.repeat(next_legal_actions.shape[0], axis=0)

                optimal_actions = self.q_target([optimal_states, next_legal_actions])

                next_action = self.policy_greedy.sample(optimal_actions)

                optimal_action = np.zeros((1, 9))

                optimal_action[:, legal_idx] = next_action

                target_q_value = reward + self.discount_factor * self.q([next_state, optimal_action])
            else:
                target_q_value = reward
            # compute td error
            with GradientTape() as tape:
                q_value = self.q([state, action])
                q_loss = mean_squared_error(target_q_value, q_value)

            self.loss_hist.append(q_loss)

            grads = tape.gradient(q_loss, self.q.trainable_variables)
            self.q_optimizer.apply_gradients(zip(grads, self.q.trainable_variables))

            # apply soft update
            grads = [(b - a) for a, b in zip(self.q.trainable_variables, self.q_target.trainable_variables)]
            self.q_target_optimizer.apply_gradients(zip(grads, self.q_target.trainable_variables))

        else:
            print('Didn\'t update because no samples available.')

        return

    def save_model(self):

        self.q.save(f'./models/tictactoe/{self.id}/q')
        self.q_target.save(f'./models/tictactoe/{self.id}/q_target')

        return




