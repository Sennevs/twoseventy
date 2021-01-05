from ..components import Candidate
from ..data import candidate_data

import random
import string

# set constants
BASE_BUDGET = 250


class Player:

    def __init__(self, name, candidate):

        self.id = f'{name}_{"".join(random.choice(string.ascii_uppercase) for _ in range(6))}'
        self.name = name
        self.candidate = Candidate(**candidate_data[candidate])

        self.national_groups = []
        self.state_groups = []

        self.budget = BASE_BUDGET
        self.spend_budget = 0

    def set_budget(self, budget):

        self.budget = budget

        print(f'New budget for {self.name} is {self.budget}.')

        return self.budget
