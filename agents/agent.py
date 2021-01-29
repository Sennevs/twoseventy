import json
import tkinter as tk

from tensorflow import GradientTape
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam, SGD

from .q_network import QNetwork
from .policies import EGreedy, Greedy
from .replay_buffer import ReplayBuffer

BATCH_SIZE = 1


class Agent:

    def __init__(self, agent_id):

        self.id = agent_id
        self.player = None

        return

    def choose_player(self):
        raise NotImplementedError('Define a choose_player function for your agent first.')

    def play(self, env):
        raise NotImplementedError('Define a play function for your agent first.')


class AI(Agent):

    def __init__(self, agent_id):
        super().__init__(agent_id)

        # this part needs to be rewritten, so we can dynamically choose the config params

        self.q = QNetwork()
        self.q_target = QNetwork()

        self.discount_factor = 0.99

        # make policy variable in the future
        self.policy = EGreedy(epsilon=0.05)
        self.policy_greedy = Greedy()

        # not sure if this should be list or numpy/tensor
        # this is what we need to keep track of the environment to let our agent reflect and learn from its interactions
        self.observed_states = []
        self.previous_actions = []
        self.observed_rewards = []
        self.previous_action_spaces = []

        # replay buffer still needs to be implemented and is currently not in use
        self.replay_buffer = ReplayBuffer()
        self.use_rb = False

        self.q_optimizer = Adam()
        self.q_target_optimizer = SGD()

    def play(self, env):

        playing = False
        optimal_actions = []
        while playing:
            # get legal actions from simulator
            legal_actions = env.get_legal_actions(self.id)

            # loop over legal actions and select action with highest q-value\
            state = env.view_board()
            optimal_action = self.q_target(state, legal_actions)
            optimal_actions.append(optimal_action)

            # input optimal action in env, get state back and playing variable,
            # playing variable is always true except if input is end_turn
            state, playing = env.play_action(optimal_action, self.id)

        self.previous_actions.append(optimal_actions)
        return

    def observe(self, env):

        # need to implement reward mechanism here
        reward = 0
        state = 0

        self.observed_states.append(state)
        self.observed_rewards.append(reward)

        return

    def choose_player(self):

        self.player = {'player': self.id, 'candidate': 'BDR'}

        return self.player

    def update(self):


        # should find a way to make the update rule variable
        state, action, reward, next_state, action_space = self.replay_buffer.sample(size=BATCH_SIZE)

        next_action = self.policy_greedy.sample(self.q_target, state, action_space)

        # compute td error
        with GradientTape() as tape:
            q_value = self.q(state, action)
            target_q_value = reward + self.discount_factor * self.q(next_state, next_action)
            q_loss = mean_squared_error(q_value, target_q_value)

        grads = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.q.trainable_variables))

        # apply soft update
        grads = [(b - a) for a, b in zip(self.q.trainable_variables, self.q_target.trainable_variables)]
        self.q_target_optimizer.apply_gradients(zip(grads, self.q_target.trainable_variables))

        return


class Human(Agent):

    def __init__(self, agent_id, ui):

        super().__init__(agent_id)

        self.ui = ui
        self.action_ui_function = self._select_action_ui()
        self.actions_taken = 0

    def choose_player(self):

        if self.ui is None:
            ans = self._choose_player_no_ui()
        elif self.ui == 'text':
            ans = self._choose_player_text_ui()
        elif self.ui == 'fields':
            ans = self._choose_player_fields_ui()
        else:
            raise ValueError(f'{self.ui.capitalize()} is currently not implemented. '
                             f'Please choose either None, text or fields as your ui.')

        return ans

    def play(self, env):

        actions = self.action_ui_function()

        return actions

    def _select_action_ui(self):

        if self.ui is None:
            ans = self._action_no_ui
        elif self.ui == 'text':
            ans = self._action_text_ui
        elif self.ui == 'fields':
            ans = self._action_fields_ui
        else:
            raise ValueError(f'{self.ui.capitalize()} is currently not implemented. '
                             f'Please choose either None, text or fields as your ui.')

        return ans

    def _choose_player_no_ui(self):

        actions = [{'name': 'Senne', 'candidate': 'BD', 'type': 'ai'},
                   {'name': 'Barry', 'candidate': 'MPR', 'type': 'human'}]

        return actions

    def _choose_player_text_ui(self):

        raise NotImplementedError('This part hasn\'t been implemented yet.')

    def _choose_player_fields_ui(self):

        raise NotImplementedError('This part hasn\'t been implemented yet.')

    def _action_no_ui(self):

        if self.actions_taken % 3:
            actions = [{'state': {'NY': 1}, 'ng': {'BC': 1}}, {'state': {'FL': 2}, 'ng': {'YV': 1}}]
        elif (self.actions_taken + 1) % 3:
            actions = [{'state': {'NY': 1}, 'ng': {'BC': 1}}, {'state': {'NC': 2, 'PA': 2}, 'ng': {}}]
        else:
            actions = [{'state': {'NY': 1}, 'ng': {'BC': 1}}, {'state': {'FL': 11, 'NC': 1, 'PA': 1}, 'ng': {}}]

        return actions

    def _action_text_ui(self):

        actions = []
        for player in self.env.players:
            action = input(f'Please input action for {player.name} ')
            action = json.loads(action)
            actions.append(action)

        return actions

    def _action_fields_ui(self):

        raise NotImplementedError('The fields ui has not been implemented yet.')

        fields = 'Last Name', 'First Name', 'Job', 'Country'

        def fetch(entries):
            for entry in entries:
                field = entry[0]
                text = entry[1].get()
                print('%s: "%s"' % (field, text))

        def makeform(root, fields):
            entries = []
            for field in fields:
                row = tk.Frame(root)
                lab = tk.Label(row, width=15, text=field, anchor='w')
                ent = tk.Entry(row)
                row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
                lab.pack(side=tk.LEFT)
                ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
                entries.append((field, ent))
            return entries

        # if __name__ == '__main__':
        root = tk.Tk()
        ents = makeform(root, fields)
        root.bind('<Return>', (lambda event, e=ents: fetch(e)))
        b1 = tk.Button(root, text='Show',
                       command=(lambda e=ents: fetch(e)))
        b1.pack(side=tk.LEFT, padx=5, pady=5)
        b2 = tk.Button(root, text='Quit', command=root.quit)
        b2.pack(side=tk.LEFT, padx=5, pady=5)
        root.mainloop()

        return actions
