'''Experiment: how much the final answer depends on max_k.

Methods 4 and 5 scale the plot x axis by max_k, so their answer can change when max_k
changes. Measured on the datasets with separated clusters, one seed.

    python3 experiments/max_k_sensitivity.py
'''

from common import finder_for, is_clear, suite, vote

MAX_KS = (10, 20, 30, 40)
SEED = 1


def main():
    print(f'{"dataset":24}{"truth":>6}' + ''.join(f'{m:>7}' for m in MAX_KS))
    clear = changed = 0
    for name, X, truth, separation in suite():
        if X.shape[0] < 30 or not is_clear(separation):
            continue
        answers = []
        for max_k in MAX_KS:
            finder = finder_for(X, SEED, max_k=max_k)
            rows = finder.stats
            if not rows:
                answers.append(None)
                continue
            values = [finder._method1(rows), finder._method2(rows), finder._method3(rows),
                      finder._method4(rows), finder._method5(rows)]
            answers.append(vote(values))
        clear += 1
        changed += len(set(answers)) > 1
        print(f'{name:24}{truth:>6}' + ''.join(f'{str(a):>7}' for a in answers))

    print(f'\nclear datasets: {clear}, answer changes across max_k in {MAX_KS}: {changed}')
    print(f'seed: {SEED}, datasets with fewer than 30 samples are excluded')


if __name__ == '__main__':
    main()
