import numpy as np
from random import shuffle
from collections import deque
from games import Env


class Player:

    def __init__(self, player_id):

        self.observed_state = None
        self.id = player_id

        return


class TicTacToeEnv(Env):

    def __init__(self, players, board_size=3, start_order='test'):

        # temp checks because not everything is supported yet
        if len(players) != 2:
            raise ValueError('Currently only 2 players is supported.')

        if board_size != 3:
            raise ValueError('Currently only 2 players is supported.')

        if len(set(players)) != len(players):
            raise ValueError('Player names should be unique.')

        self.players = {player: Player(player) for player in players}
        self.board_size = board_size
        self.start_order = start_order

        self.board = None
        self._turn_sequence = None
        self._turn_starts = None
        self.active_player = None
        self.info = {}

        self.done = False
        self.winner = None
        self.rewards = {player: None for player in self.players}
        return

    def visualize_board(self):

        ans = self.board * list(range(1, len(self.players) + 1))
        ans = ans.sum(axis=1).reshape((3, 3))

        return ans

    def update_active_player(self):

        self.active_player = self._turn_sequence.popleft()
        self._turn_sequence.append(self.active_player)

        return

    def reset_board(self):

        self.board = np.zeros((self.board_size * self.board_size, len(self.players)))

        return

    def reset(self):

        self.initialize_turn_sequence()
        self.update_active_player()
        self.reset_board()
        self.info = {}
        self.rewards = {player: None for player in self.players}

        self.winner = None

        self.done = False

        # print(f'Player {self.active_player} starts.')

        return self.active_player

    def initialize_turn_sequence(self):

        if self.start_order == 'random':
            self._turn_sequence = deque(self.players.keys())
            shuffle(self._turn_sequence)
        elif self.start_order == 'test':
            self._turn_sequence = deque(self.players.keys())
        else:
            self._turn_sequence = deque(self.start_order)

        self._turn_starts = self._turn_sequence.copy()

        return

    def step(self, player, action):

        #import time

        # check if this player is allowed to make a move right now
        #start_time = time.time()
        self.verify_player(player)

        #print(f'Verifying player took {time.time() - start_time} seconds')
        #start_time = time.time()

        # check if move is legal
        self.verify_action(action)
        #print(f'Verifying action took {time.time() - start_time} seconds')
        #start_time = time.time()

        # update game state
        self.confirm_move(action)
        #print(f'Confirming move took {time.time() - start_time} seconds')
        #start_time = time.time()

        # check if there's a winner
        self.check_win()
        #print(f'Checking win took {time.time() - start_time} seconds')
        #start_time = time.time()

        # update active player
        self.update_active_player()
        #print(f'Updating active player took {time.time() - start_time} seconds')
        #start_time = time.time()

        self.rewards = self._update_rewards()
        #print(f'Updating rewards took {time.time() - start_time} seconds')
        #start_time = time.time()

        return self.done, self.active_player, self.info

    def verify_player(self, player):

        if player != self.active_player:
            raise ValueError('This action is illegal because a different player is currently required to make a move.')

        return

    def verify_action(self, action):

        # check if only one action is executed
        if (sum(action == 0) != (self.board_size ** 2 - 1)) or (sum(action == 1) != 1):
            raise ValueError('Action is illegal: more than 1 action is performed.')

        if self.board[action == 1].sum() != 0:
            raise ValueError('Action is illegal: tile is already occupied.')

        return

    def get_legal_actions(self):

        ans = ~self.board.sum(axis=1).astype(bool).reshape((1, -1))

        return ans

    def observe(self, player):

        legal_actions = self.get_legal_actions()

        done = self.done


        return self.board.transpose(1,0).reshape(1, -1), np.array([self.rewards[player]]), legal_actions, np.array([done])

    def check_win(self):

        eval_board = self.board[:, self._turn_starts.index(self.active_player)].reshape([self.board_size,
                                                                                         self.board_size])

        rows = eval_board.sum(axis=0)
        cols = eval_board.sum(axis=1)
        diag = np.fliplr(eval_board).diagonal().sum(axis=0).reshape([1])
        adiag = eval_board.diagonal().sum(axis=0).reshape([1])

        res = np.concatenate([rows, cols, diag, adiag], axis=0)

        if 3 in res:
            self.winner = self.active_player
            self.done = True

        if not np.any(self.board.sum(axis=1) == 0):
            self.done = True
            self.winner = None

        return self.winner

    def confirm_move(self, action):

        self.board[:, self._turn_starts.index(self.active_player)] += action
        #self.board[:, list(self.players.keys()).index(self.active_player)] += action

        return

    def _update_rewards(self):

        if self.done and self.winner is not None:
            rewards = {player: 1 if (player == self.winner) else -1 for player in self.players}
        else:
            rewards = {player: 0 for player in self.players}

        return rewards

    def close(self):

        return
