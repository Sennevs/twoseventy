import string
import random


class Candidate:

    def __init__(self, id, name, party, state_bonuses, national_bonuses):

        self.id = f'{id}_{"".join(random.choice(string.ascii_uppercase) for _ in range(6))}'
        self.name = name
        self.party = party
        self.state_bonuses = state_bonuses
        self.national_bonuses = national_bonuses

        self.state_groups = []
        self.national_groups = []

        return


