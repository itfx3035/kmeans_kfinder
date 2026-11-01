'''Experiment: the reporting convention (raw k vs elbow vertex k - 1).

Method 1 reports the raw k of the winning row, methods 2..5 report k - 1, so the vote
mixes two conventions. This measures which convention agrees more often with the known
ground truth k per method, and what the vote would do if every method used the same
convention. Only datasets with separated clusters are counted, because there the ground
truth k is what an elbow method should recover.

    python3 experiments/convention.py
'''

from common import SEEDS, finder_for, is_clear, suite, vote

METHODS = ('method1', 'method2', 'method3', 'method4', 'method5')


def as_convention(value, method_index, raw):
    '''Re-express a method result in the requested convention.

    Method 1 selects a row (raw k); methods 2..5 select an elbow vertex (k - 1).
    """raw=True""" means report the raw k, raw=False means report the vertex.
    '''
    if value is None:
        return None
    own_is_raw = method_index == 0
    if raw == own_is_raw:
        return value
    return value - 1 if own_is_raw else value + 1


def main():
    runs = 0
    own_hits = dict.fromkeys(METHODS, 0)
    opposite_hits = dict.fromkeys(METHODS, 0)
    vote_own = vote_raw = changed = 0

    print(f'{"method":9}{"own convention":>16}{"opposite":>11}')
    for name, X, truth, separation in suite():
        if X.shape[0] < 30 or not is_clear(separation):
            continue
        for seed in SEEDS:
            finder = finder_for(X, seed)
            values = [finder._method1(finder.stats), finder._method2(finder.stats),
                      finder._method3(finder.stats), finder._method4(finder.stats),
                      finder._method5(finder.stats)]
            if not finder.stats:
                continue
            runs += 1
            for i, value in enumerate(values):
                own_hits[METHODS[i]] += value == truth
                opposite_hits[METHODS[i]] += as_convention(value, i, raw=(i != 0)) == truth

            current = vote(values)
            uniform = vote([as_convention(v, i, True) for i, v in enumerate(values)])
            vote_own += current == truth
            vote_raw += uniform == truth
            changed += current != uniform

    for method in METHODS:
        print(f'{method:9}{own_hits[method]:>16}/{runs}{opposite_hits[method]:>11}/{runs}')

    print(f'\nvote with the current mixed convention: {vote_own}/{runs}')
    print(f'vote with every method reported as raw k: {vote_raw}/{runs}')
    print(f'runs where the convention changes the answer: {changed}/{runs}')
    print(f'runs: {runs} (clear datasets only, separation >= {2.0})')


if __name__ == '__main__':
    main()
