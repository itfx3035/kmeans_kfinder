'''Before/after measurement harness for kmeans_kfinder.

Runs every sample dataset through two implementations and prints the same metrics
for both, so improvements can be measured.

    python3 measure.py baseline.kmeans_kfinder_baseline kmeans_kfinder

Exit code is 0 when the last implementation under test meets every documented
behaviour check, 1 otherwise.
'''

import argparse
import importlib
import sys
import time
import warnings

from sample_dataset import DATASETS, get_dataset

CHECKS = ['crash_free', 'vote_follows_rule', 'fit_best_works', 'stable_across_seeds',
          'kwargs_not_mutated']


def documented_vote(opts):
    '''Majority vote, largest k wins ties (documented rule).'''
    values = [v for v in opts.values() if v is not None]
    if not values:
        return None
    counts = {v: values.count(v) for v in values}
    top = max(counts.values())
    return max(v for v, c in counts.items() if c == top)


def label_for(improved, same):
    '''IMPROVED / same / REGRESSION for one metric comparison.'''
    if same:
        return 'same'
    return 'IMPROVED' if improved else 'REGRESSION'


def run_module(module_name):
    mod = importlib.import_module(module_name)
    KFinder = mod.KMeansKFinder
    rows = []

    for name, spec in DATASETS.items():
        X, truth = get_dataset(name)
        rec = {'dataset': name, 'truth': truth}

        # 1. plain run, runtime and warnings measured
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                finder = KFinder(X, max_k=30, random_state=1)
                best_k = finder.find_best_k()
            rec.update({
                'best_k': best_k,
                'opts': finder.best_k_opts,
                'agree_with_truth': best_k == truth,
                'vote_follows_rule': best_k == documented_vote(finder.best_k_opts),
                'warning_count': len(caught),
                'crash_free': True,
            })
        except Exception as exc:  # noqa: measurement must record crashes
            rec.update({'best_k': None, 'opts': {}, 'agree_with_truth': False,
                        'vote_follows_rule': False, 'warning_count': 0,
                        'crash_free': False, 'error': f'{type(exc).__name__}: {exc}'})

        # 2. fit_best() without calling find_best_k first
        try:
            km = KFinder(X, max_k=30, random_state=1).fit_best()
            rec['fit_best_works'] = True
            rec['fit_best_fitted_k'] = int(km.n_clusters)
        except Exception as exc:
            rec['fit_best_works'] = False
            rec['fit_best_error'] = type(exc).__name__

        # 3. stability across 5 random seeds
        seeds = []
        for seed in range(1, 6):
            try:
                seeds.append(KFinder(X, max_k=30, random_state=seed).find_best_k())
            except Exception as exc:
                seeds.append(f'{type(exc).__name__}')
        rec['seeds'] = seeds
        seeds_ok = not isinstance(seeds[0], str)
        rec['stable_across_seeds'] = (len(set(seeds)) == 1) if seeds_ok else 'n/a'

        # 4. caller kwargs must not be mutated
        caller_kwargs = {'random_state': 1, 'n_init': 3}
        try:
            KFinder(X, max_k=5, **caller_kwargs).find_best_k()
            rec['kwargs_not_mutated'] = caller_kwargs == {'random_state': 1, 'n_init': 3}
        except Exception:
            rec['kwargs_not_mutated'] = False

        # 5. runtime for max_k=30 (only the work the implementation actually does)
        try:
            t0 = time.perf_counter()
            finder = KFinder(X, max_k=30, random_state=1)
            finder.find_best_k()
            rec['time_ms'] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception:
            rec['time_ms'] = None

        rows.append(rec)
    return rows


def print_report(rows, module_name):
    print(f'\n=== {module_name} ===')
    print(f'{"dataset":22}{"truth":>6}{"best_k":>7}{"agree":>7}'
          + ''.join(f'{c[:12]:>18}' for c in CHECKS) + f'{"warn":>6}{"ms":>8}  notes')
    for r in rows:
        line = (f'{r["dataset"]:22}{r["truth"]:>6}{str(r["best_k"]):>7}'
                f'{str(r["agree_with_truth"]):>7}')
        for check in CHECKS:
            line += f'{str(r[check]):>18}'
        notes = r.get('error') or r.get('fit_best_error') or ''
        print(line + f'{r["warning_count"]:>6}{str(r["time_ms"]):>8} {notes}')

    score = {}
    for check in CHECKS:
        measurable = [r for r in rows if r[check] != 'n/a']
        score[check] = f"{sum(bool(r[check]) for r in measurable)}/{len(measurable)}"
    score['agree_with_truth'] = sum(r['agree_with_truth'] for r in rows)
    score['total_time_ms'] = round(sum(r['time_ms'] or 0 for r in rows), 1)
    print('score: ' + ' '.join(f'{k}={v}' for k, v in score.items()))
    return score


def main():
    parser = argparse.ArgumentParser(
        description='Measure implementations against the sample datasets.')
    parser.add_argument('modules', nargs='+')
    args = parser.parse_args()

    reports = []
    for module_name in args.modules:
        rows = run_module(module_name)
        reports.append((module_name, rows, print_report(rows, module_name)))

    if len(reports) > 1:
        base_rows, new_rows = reports[0][1], reports[-1][1]
        base_score, new_score = reports[0][2], reports[-1][2]
        print('\n=== delta (improved - baseline) ===')
        for key in new_score:
            base_val, new_val = base_score[key], new_score[key]
            if key == 'total_time_ms':
                label = label_for(new_val < base_val, new_val == base_val)
            elif isinstance(base_val, str):
                b_num, b_den = (int(x) for x in base_val.split('/'))
                n_num, n_den = (int(x) for x in new_val.split('/'))
                if b_den != n_den:
                    label = 'new coverage'
                else:
                    label = label_for(n_num > b_num, n_num == b_num)
            else:
                label = label_for(new_val > base_val, new_val == base_val)
            print(f'{key:20} {base_val} -> {new_val}  {label}')
        for check in CHECKS:
            diffs = [(b['dataset'], b[check], n[check]) for b, n in zip(base_rows, new_rows)
                     if b[check] != n[check]]
            if diffs:
                print(f'  {check}: ' + ', '.join(f'{d} [{a} -> {b}]' for d, a, b in diffs))

        comparable = [(b, n) for b, n in zip(base_rows, new_rows) if b['time_ms'] and n['time_ms']]
        base_time = round(sum(b['time_ms'] for b, _ in comparable), 1)
        new_time = round(sum(n['time_ms'] for _, n in comparable), 1)
        print(f'{"runtime on datasets both ran":20} {base_time} -> {new_time} ms '
              f'({"IMPROVED" if new_time < base_time else "same"})')

        print('\nbehaviour preserved on datasets where the baseline produced an answer:')
        for b, n in zip(base_rows, new_rows):
            if b['best_k'] is None:
                continue
            status = ('same answer' if b['best_k'] == n['best_k']
                      else f'changed {b["best_k"]} -> {n["best_k"]}')
            print(f'  {b["dataset"]:22} {status}')

    last_scores = reports[-1][2]
    ok = all(v.split('/')[0] == v.split('/')[1] for v in last_scores.values() if isinstance(v, str))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
