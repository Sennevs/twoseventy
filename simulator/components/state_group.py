from ..data import state_data


class StateGroup:

    def __init__(self, name, extra_funds, states, players, id=None):

        self.id = id or name
        self.name = name
        self.extra_funds = extra_funds
        self.states = states

        self.owner = None
        self.dist = [0 for _ in range(0, players)]

        self.total_sum = sum([state['campaign_cost'] for state in state_data if state['id'] in self.states])

    def set_owner(self, owner):

        self.owner = owner

        return
