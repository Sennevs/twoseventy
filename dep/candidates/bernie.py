from .candidate import Candidate


class Bernie:

    def __init__(self):
        super().__init__(id='BD',
                         name='Bernie',
                         party='Democrat',
                         state_bonuses={'AA': 0, 'L': 0, 'OG': -25, 'HT': 25, 'A': 0, 'MB': 25, 'OS': -25, 'ED': -25,
                                        'SS': 0, 'TG': 25},
                         national_bonuses={'YV': 20, 'BC': -20, 'LE': 20, 'E': -20, 'WM': 0, 'GL': -20})

