import json
import tkinter as tk

from env.env import Env
from agents import AI, Human


class Simulator:

    def __init__(self, players):

        self.env = Env(players)
        self.turn = 0
        self.agents = {player['name']: AI(player['name']) if player['type'] == 'ai' else Human(player['name'])
                       for player in self.env.players}

    def play(self, train=False, ui=None):

        ui_function = self._select_ui(ui)

        done = False
        while not done:

            print('Current state of the board:')
            # update this to better state representation
            print(self.env.board.states)

            for player in self.env.players:
                if player.type == 'ai':
                    if train:
                        self.agents[player['name']].train()
                    else:
                        self.agents[player['name']].predict()
                    # execute
                else:
                    actions = ui_function()

            state, budgets, reward, done, info = self.env.step(actions)
            self.turn += 1

        print('The game has been finished.')

        return

    def _get_action_no_ui(self):

        if self.turn % 3:
            actions = [{'state': {'NY': 1}, 'ng': {'BC': 1}}, {'state': {'FL': 2}, 'ng': {'YV': 1}}]
        elif (self.turn + 1) % 3:
            actions = [{'state': {'NY': 1}, 'ng': {'BC': 1}}, {'state': {'NC': 2, 'PA': 2}, 'ng': {}}]
        else:
            actions = [{'state': {'NY': 1}, 'ng': {'BC': 1}}, {'state': {'FL': 11, 'NC': 1, 'PA': 1}, 'ng': {}}]

        return actions

    def _get_action_text_ui(self):

        actions = []
        for player in self.env.players:
            action = input(f'Please input action for {player.name} ')
            action = json.loads(action)
            actions.append(action)

        return actions

    def _get_action_fields_ui(self):

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
