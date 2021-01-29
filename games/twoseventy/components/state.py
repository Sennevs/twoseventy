class State:

    def __init__(self, id, name, code, electoral_votes, campaign_cost, popular_vote, groups, players):

        self.id = id
        self.name = name
        self.code = code
        self.electoral_votes = electoral_votes
        self.campaign_cost = campaign_cost
        self.popular_vote = popular_vote
        self.groups = groups

        self.current_highest = 0

        self.leader = None
        self.owner = None
        self.winner = None

        self.no_limit = set()
        self.dist = {player: 0 for player in range(0, players)}

