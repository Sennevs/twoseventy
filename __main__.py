

import argparse

from simulator import Simulator


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--players', action='store', type=str, nargs='*')
    parser.add_argument('--ui', type=str, required=False)
    args = vars(parser.parse_args())

    if args['players'] is None:
        args['players'] = [{'name': 'Senne', 'candidate': 'BD', 'type': 'ai'},
                           {'name': 'Barry', 'candidate': 'MPR', 'type': 'human'}]

    sim = Simulator(args['players'])
    sim.play(args['ui'])

    return


if __name__ == "__main__":

    main()
