'''Experiment: method 1 as a gain-versus-cost tradeoff, score = delta - c/k.

The documented formula is c = 1. The "true derivative" reading gives
delta + 1/(k(k-1)), which decays as 1/k^2 and degenerates into the largest single drop.
This grid checks whether any value of c agrees more often with the known ground truth k,
for method 1 alone and for the final vote.

    python3 experiments/method1_grid.py
'''

from common import SEEDS, finder_for, is_clear, suite, vote

C_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DERIVATIVE = 'derivative'
FORMULAS = [f'c={c}' for c in C_VALUES] + [DERIVATIVE]


def score(delta, k, formula):
    if formula == DERIVATIVE:
        return delta - (1 / k - 1 / (k - 1))
    return delta - float(formula.split('=')[1]) / k


def argmax(rows, formula):
    best_k, best = None, None
    for row in rows:
        if row.k < 2 or row.delta_distance_pct is None:
            continue
        value = score(row.delta_distance_pct, row.k, formula)
        if best is None or value > best:
            best, best_k = value, row.k
    return best_k


def main():
    runs = []
    for name, X, truth, separation in suite():
        if X.shape[0] < 30:
            continue
        for seed in SEEDS:
            finder = finder_for(X, seed)
            rows = finder.stats
            if not rows:
                continue
            others = [finder._method2(rows), finder._method3(rows),
                      finder._method4(rows), finder._method5(rows)]
            entry = {'truth': truth, 'clear': is_clear(separation)}
            for formula in FORMULAS:
                method1 = argmax(rows, formula)
                entry[formula] = (method1, vote([method1] + others))
            runs.append(entry)

    clear_runs = [r for r in runs if r['clear']]
    print(f'{"formula":12}{"method1 alone":>16}{"method1 alone (clear)":>22}'
          f'{"vote":>7}{"vote (clear)":>13}')
    for formula in FORMULAS:
        alone = sum(1 for r in runs if r[formula][0] == r['truth'])
        alone_clear = sum(1 for r in clear_runs if r[formula][0] == r['truth'])
        v = sum(1 for r in runs if r[formula][1] == r['truth'])
        v_clear = sum(1 for r in clear_runs if r[formula][1] == r['truth'])
        print(f'{formula:12}{alone:>10}/{len(runs)}{alone_clear:>16}/{len(clear_runs)}'
              f'{v:>5}/{len(runs)}{v_clear:>10}/{len(clear_runs)}')
    print(f'\nruns: {len(runs)} ({len(clear_runs)} clear, separation >= 2.0)')


if __name__ == '__main__':
    main()
