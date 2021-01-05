from .candidate import Candidate


class McCain:

    def __init__(self):
        super().__init__(id='MR',
                         name='McCain',
                         party='Republican',
                         state_bonuses={'AA': -10, 'L': -15, 'OG': 20, 'HT': 10, 'A': 10, 'MB': 10, 'OS': -15, 'ED': 10,
                                        'SS': 25, 'TG': 0},
                         national_bonuses={'YV': -10, 'BC': 20, 'LE': 10, 'E': 10, 'WM': 0, 'GL': 0})
