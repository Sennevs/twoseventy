from .components import Board, Player

# set constants
BASE_BUDGET = 250
MAX_STEPS = 10


class Env:

    def __init__(self, players):

        self.base_budget = BASE_BUDGET
        self.max_steps = MAX_STEPS

        self.players = [Player(**player) for player in players]
        self.board = Board(len(self.players))

        self.budgets = [player.budget for player in self.players]
        self.final_round = False

        self.current_step = 0

        self.info = {}
        self.state = []
        self.rewards = {}
        self.done = False

    def get_state(self, player_id):

        return self.state[player_id]

    def step(self, actions):

        # update current step
        self.current_step += 1

        # calculate spend budget for each player
        self.calculate_spend_budget(actions)

        # verify if the actions are legal
        self.verify_action_legality(actions)

        # update board state
        state_actions = [action['state'] for action in actions]
        ng_actions = [action['ng'] for action in actions]
        self.board.update_states(state_actions)
        self.board.update_state_groups()
        self.board.update_national_groups(ng_actions)

        # update player budgets
        self.budgets = self.update_budget()

        # check if final_round
        if self.final_round:
            # count electoral votes for each player and return rewards
            self.rewards = self.board.count_ballots()
            self.done = True
        else:
            # check if final_round next turn
            self.rewards = []
            self.final_round = self.board.check_ballot()

        for key, val in self.board.state_groups.items():
            print(f'Current owner of state group {key}: {val.owner}.')

        print(self.done)
        return self.state, self.budgets, self.rewards, self.done, self.info

    def get_legal_actions(self, player):

        # check legality of state actions
        actions = []
        for key, value in self.board.states.items():
            for i in range(0, self.max_steps):
                if value.campaign_cost * i <= player.budget:
                    actions.append(1)
                else:
                    actions.append(0)

        # check legality of national group actions
        for key, value in self.board.national_groups.items():
            for i in range(0, self.max_steps):
                if value.campaign_cost * i <= player.budget:
                    actions.append(1)
                else:
                    actions.append(0)

        # add end turn move
        actions.append(1)

        return actions

    def reset(self):

        # this should be rewritten to proper level of abstraction
        self.base_budget = BASE_BUDGET

        for player in self.players:

            player.national_groups = []
            player.state_groups = []
            player.budget = BASE_BUDGET
            player.spend_budget = 0

        self.board = Board(len(self.players))

        self.budgets = [player.budget for player in self.players]
        self.final_round = False

        self.current_step = 0

        self.info = {}
        self.state = []
        self.rewards = {}
        self.done = False

        return

    def update_budget(self):

        budgets = []

        for player in self.players:
            player.budget -= player.spend_budget
            player.budget += BASE_BUDGET
            for key, value in self.board.national_groups.items():

                player.budget += value.extra_funds * (1 + player.candidate.national_bonuses[key]/100)

            for key, value in self.board.state_groups.items():
                player.budget += value.extra_funds * (1 + player.candidate.state_bonuses[key]/100)

            budgets.append(player.budget)

        return budgets

    def verify_action_legality(self, actions):

        # check if entered actions are allowed given the current state

        self._verify_action_size(actions)
        self._verify_budget()
        self._verify_max_invest(actions)
        self._verify_state_entry(actions)

    def calculate_spend_budget(self, actions):

        spend_budgets = []
        for player, val in enumerate(actions):
            state_action = val['state']
            ng_action = val['ng']
            spend_budget = 0
            spend_budget += sum([v*self.board.states[k].campaign_cost for k, v in state_action.items()])
            spend_budget += sum([v*self.board.national_groups[k].campaign_cost for k, v in ng_action.items()])

            self.players[player].spend_budget = spend_budget
            spend_budgets.append(spend_budget)

        return

    def _verify_action_size(self, actions):

        if len(actions) != self.board.num_players:
            message = f'Expected actions for {self.board.num_players} players, but received only {len(actions)} instead'
            raise ValueError(message)

        return

    def _verify_budget(self):

        illegal_budgets = {}
        for player_idx, player in enumerate(self.players):
            if player.spend_budget > player.budget:
                illegal_budgets[player.name] = {'allowed_budget': player.budget, 'action_budget': player.spend_budget}

        if len(illegal_budgets) > 0:
            raise ValueError(f'At least 1 player spent more than he or she was allowed: {illegal_budgets}')

        return

    def _verify_max_invest(self, actions):

        illegal_moves = {}
        for (player_idx, player), action in zip(enumerate(self.players), actions):
            state_action = action['state']
            ng_action = action['ng']

            illegal_state_move = {key: {'old_value': self.board.states[key].dist[player_idx],
                                        'action': value,
                                        'new_value': value + self.board.states[key].dist[player_idx]}
                                  for key, value in state_action.items()
                                  if value + self.board.states[key].dist[player_idx] > 10}

            illegal_ng_move = {key: {'old_value': self.board.national_groups[key].dist[player_idx],
                                     'action': value,
                                     'new_value': value + self.board.national_groups[key].dist[player_idx]}
                               for key, value in ng_action.items()
                               if value + self.board.national_groups[key].dist[player_idx] > 10}

            if len(illegal_state_move) > 0 or len(illegal_ng_move) > 0:
                illegal_moves[player.name] = {'state': illegal_state_move, 'ng': illegal_ng_move}

        if len(illegal_moves) > 0:
            raise ValueError(f'At least 1 player made an illegal move: {illegal_moves}')

        return

    def _verify_state_entry(self, actions):

        pass
