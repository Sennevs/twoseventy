
# set constants
BASE_BUDGET = 250

class Candidate:

    def __init__(self, id, name, party, state_bonuses, national_bonuses):

        self.id = id
        self.name = name
        self.party = party
        self.state_bonuses = state_bonuses
        self.national_bonuses = national_bonuses

        self.state_groups = []
        self.national_groups = []

        self.budget = BASE_BUDGET

    def spend_budget(self, ):

        pass

    def fund_budget(self):

        self.budget += BASE_BUDGET
        for national_group in self.national_groups:
            self.budget += national_group.extra_funds * (1 + self.national_bonuses[national_group]/100)

        for state_group in self.state_groups:
            self.budget += state_group.extra_funds * (1 + self.state_bonuses[state_group]/100)


        print(f'New budget for {player.name} is {player.budget}.')

        return
