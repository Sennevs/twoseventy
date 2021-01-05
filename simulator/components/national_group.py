class NationalGroup:

    def __init__(self, id, name, extra_funds, campaign_cost, players):

        self.id = id
        self.name = name
        self.extra_funds = extra_funds
        self.campaign_cost = campaign_cost

        self.owner = None
        self.dist = {key: 0 for key in range(0, players)}

        self.leader = None
        self.owner = None
        self.winner = None

        self.current_highest = 0
        self.no_limit = set()

        return

    def update(self, group, value):

        self.dist[group] += value

        return

    def update_owners(self, board_state):

        pass
