from .national_group import NationalGroup
from .state import State
from .state_group import StateGroup
from ..data import national_group_data, state_data, state_group_data


class Board:

    def __init__(self, num_players):

        self.national_groups = {national_group['id']: NationalGroup(**national_group, players=num_players)
                                for national_group in national_group_data}
        self.states = {state['id']: State(**state, players=num_players) for state in state_data}
        self.state_groups = {state_group['id']: StateGroup(**state_group, players=num_players)
                             for state_group in state_group_data}

        self.num_players = num_players
        self.owners = {}

        return

    def update_states(self, actions):

        self._update_areas(actions, self.states)

        return

    def update_national_groups(self, actions):

        self._update_areas(actions, self.national_groups)

        return

    def update_state_groups(self):

        for _, val in self.state_groups.items():

            val.dist = [0 for _ in range(0, len(val.dist))]
            for state in val.states:
                if self.owners[state] is not None:
                    val.dist[self.owners[state]] += self.states[state].campaign_cost
            max_val = max(val.dist)
            if max_val > (val.total_sum - max_val):
                val.owner = val.dist.index(max_val)

            print(f'Distribution for {_} is: {val.dist}')

        return

    def check_ballot(self):

        # check if there's a candidate who can claim credit for every state or state has been campaigned in
        # and conflicted out

        # check if states that are won add up to at least 100

        pass

    def count_ballots(self):

        # for each candidate, count the lead states electoral votes, return list with overview of votes per candidate
        # and candidate winner

        pass

    def _update_areas(self, actions, areas):

        # check if any players invest in the same state
        union_set = set()
        intersect_set = set()
        for action in actions:
            intersect_set = intersect_set.union(union_set.intersection(action.keys()))
            union_set = union_set.union(action.keys())

        # check if there are players that have a tie after investing in the same state
        competing_areas = {key: {} for key in intersect_set}
        for area in intersect_set:
            for player, action in enumerate(actions):
                if area in action:
                    votes = str(areas[area]['values'][player] + action[area])
                    if votes not in areas[area]:
                        competing_areas[area][votes] = []
                    competing_areas[area][votes].append(player)

        # create filter for states that clash
        filtered_compete = {key: set() for key in list(range(0, len(actions)))}
        for area in competing_areas:
            for l in competing_areas[area]:
                for a in competing_areas[area][l]:
                    filtered_compete[a].add(area)

        # update values for each state, using filter for clashes
        for player, action in enumerate(actions):
            for key, value in action.items():
                if key not in filtered_compete[player]:
                    areas[key].dist[player] += value
                    if areas[key].dist[player] > areas[key].current_highest:
                        areas[key].current_highest = areas[key].dist[player]
                        areas[key].leader = player
                        if areas[key].dist[player] >= 3:
                            areas[key].owner = player
                        if areas[key].dist[player] == 10:
                            areas[key].winner = player

                    areas[key].no_limit.add(player)

        self.owners = {key: value.owner for key, value in areas.items()}

        print({key: value.dist for key, value in areas.items()})

        return
