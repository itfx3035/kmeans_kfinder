'''Experiment: does changing method 1 improve the final answer?

Variants of the method 1 score (distance improvement vs complexity cost term):

  documented   : delta - 1/k                       (what the README documents)
  derivative   : delta - (1/k - 1/(k-1))           (a true derivative of complexity)
  ratio        : delta / (1/(k(k-1)))              (gain to cost ratio)
  cost_penalty : delta - 2/k                       (documented, doubled penalty)
  vertex       : documented, reported at the elbow vertex (k-1), the convention used
                by methods 2..5

Methods 2..5 come from the current implementation, so only method 1 differs.

    python3 experiments/method1_variants.py
'''

from common import SEEDS, is_clear, suite, vote, finder_for

VARIANTS = ['documented', 'derivative', 'ratio', 'cost_penalty', 'vertex']


def score(delta, k, variant):
    if variant in ('documented', 'vertex'):
        return delta - 1 / k
    if variant == 'derivative':
        return delta - (1 / k - 1 / (k - 1))
    if variant == 'ratio':
        return delta / (1 / (k * (k - 1)))
    if variant == 'cost_penalty':
        return delta - 2 / k
    raise ValueError(f'unknown variant {variant!r}')


def argmax(rows, variant):
    best_k, best = None, None
    for row in rows:
        if row.k < 2 or row.delta_distance_pct is None:
            continue
        value = score(row.delta_distance_pct, row.k, variant)
        if best is None or value > best:
            best, best_k = value, row.k
    if best_k is None:
        return None
    return best_k - 1 if variant == 'vertex' else best_k


def main():
    datasets = suite()
    runs = clear_runs = 0
    hits = dict.fromkeys(VARIANTS, 0)
    clear_hits = dict.fromkeys(VARIANTS, 0)
    method1_hits = dict.fromkeys(VARIANTS, 0)
    decisive = dict.fromkeys(VARIANTS, 0)
    value_changes = 0
    final_changes = 0
    picks_two = 0
    equals_largest_drop = 0

    print(f'{"dataset":24}{"truth":>6}' + ''.join(f'{v[:12]:>13}' for v in VARIANTS))
    for name, X, truth, separation in datasets:
        first = {}
        for seed in SEEDS:
            finder = finder_for(X, seed)
            rows = finder.stats
            if not rows:
                continue
            others = [finder._method2(rows), finder._method3(rows),
                      finder._method4(rows), finder._method5(rows)]
            m1 = {v: argmax(rows, v) for v in VARIANTS}

            for variant in VARIANTS:
                final = vote([m1[variant]] + others)
                if seed == SEEDS[0]:
                    first[variant] = final
                    hits[variant] += final == truth
                    if is_clear(separation):
                        clear_hits[variant] += final == truth
                method1_hits[variant] += m1[variant] == truth
                if final == m1[variant]:
                    decisive[variant] += 1

            drops = [(row.k, row.delta_distance_pct) for row in rows
                     if row.delta_distance_pct is not None]
            largest_drop_k = max(drops, key=lambda kv: kv[1])[0] if drops else None
            if m1['derivative'] == 2:
                picks_two += 1
            if m1['derivative'] == largest_drop_k:
                equals_largest_drop += 1

            if m1['documented'] != m1['derivative']:
                value_changes += 1
                if vote([m1['documented']] + others) != vote([m1['derivative']] + others):
                    final_changes += 1

            runs += 1
            if is_clear(separation):
                clear_runs += 1

        print(f'{name:24}{truth:>6}' + ''.join(f'{str(first[v]):>13}' for v in VARIANTS))

    def line(values, total):
        return ', '.join(f'{v} {values[v]}/{total}' for v in VARIANTS)

    print(f'\nfinal answer matches ground truth (first seed): {line(hits, len(datasets))}')
    print(f'final answer matches ground truth (clear datasets, first seed): '
          f'{line(clear_hits, sum(1 for d in datasets if is_clear(d[3])))}')
    print(f'method 1 value alone matches ground truth (all runs): {line(method1_hits, runs)}')
    print(f'method 1 is part of the winning majority: {line(decisive, runs)}')
    print(f'method 1 value differs documented vs derivative: {value_changes}/{runs} runs')
    print(f'final answer differs documented vs derivative: {final_changes}/{runs} runs')
    print(f'derivative variant picks k=2: {picks_two}/{runs} runs '
          f'(equals plain "largest single drop": {equals_largest_drop}/{runs})')
    print(f'runs: {runs} ({clear_runs} clear, separation >= 2.0)')


if __name__ == '__main__':
    main()
