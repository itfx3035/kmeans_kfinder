# Before / after measurement

Environment: Python 3.14.4, scikit-learn 1.9.1, numpy 2.5.4, pytest 9.1.1.
Raw machine-readable output of the harness is in `reports/comparison.txt`.

## What was measured

`sample_dataset.py` defines 7 sample datasets, each with a known ground truth `k`:

| dataset | ground truth | purpose |
|---|---|---|
| clean_blobs | 4 | well separated, all methods should agree |
| overlapping_blobs | 3 | ambiguous, elbow heuristics may disagree |
| anisotropic_blobs | 3 | large within-cluster variance, axis scaling |
| tiny_dataset | 2 | `max_k` (30) > `n_samples` (12) |
| single_cluster_noise | 1 | no real elbow, produces voting ties |
| degenerate_identical | 1 | zero total distance, division by zero |
| iris | 3 | real dataset, `n_samples` small, overlapping classes |

`tests/test_kmeans_kfinder.py` checks the documented behaviour and can run against
any implementation (`KFINDER_MODULE`), so the same suite was executed against the
pre-change code and the improved code. `measure.py` runs the same datasets through
both implementations and compares the metrics.

## Test results

| | before | after |
|---|---|---|
| passed | 8 | **16** |
| failed | 8 | **0** |

Failing before:

1. `test_fit_best_without_calling_find_best_k` — `AttributeError` (`__init__` assigned `best_k`/`best_k_opts` as local variables, never on `self`)
2. `test_voting_follows_documented_rule` — tie broken by first-seen order instead of larger k (single_cluster_noise, seed 3: `21` and `7` tie, old code returned 7)
3. `test_zero_variance_data_does_not_raise` — `ZeroDivisionError`
4. `test_degenerate_identical_points_do_not_raise` — `ZeroDivisionError`
5. `test_max_k_larger_than_n_samples_does_not_raise` — `ValueError` from sklearn (no clamping)
6. `test_max_k_one_is_handled` — returns `0` when no method produced a result
7. `test_invalid_max_k_is_rejected` — no validation
8. `test_best_model_is_cached` — refit on every `fit_best()` call, plus the `AttributeError` above

Regression guards (passed before and after, so the improvement did not change
answers that were already correct): ground truth on clean data, reproducibility on
Iris, stability on clean data, behaviour preservation against
`tests/expected_baseline.json`.

## Harness metrics

| metric | before | after |
|---|---|---|
| crash_free | 5/7 | **7/7** |
| vote follows documented rule | 5/7 | **7/7** |
| `fit_best()` works standalone | 0/7 | **6/7** |
| caller kwargs not mutated | 6/7 | **7/7** |
| stable across 5 random seeds | 2/5 measurable | 3/7 (same 2 on comparable datasets) |
| agrees with ground truth | 3/7 | 3/7 |
| runtime on datasets both ran | 1873.4 ms | 1900.4 ms (same, within noise) |

Answers on the datasets the old code could answer are identical:

```
clean_blobs          same answer
overlapping_blobs    same answer
anisotropic_blobs    same answer
single_cluster_noise same answer
iris                 same answer
```

The total time difference is only new work: the datasets that crashed before now
run to completion (tiny_dataset 90 ms, degenerate_identical 5 ms). The fair
comparison is the "runtime on datasets both ran" line, which is unchanged.
Timings vary slightly between runs, so only the counts are stable numbers.

## Changes made

| issue | fix | effect measured |
|---|---|---|
| `self.best_k` / `self.best_k_opts` never initialised | initialised in `__init__` | 8 → 0 failing tests |
| `fit_best()` refits every call | fitted model cached, best fit reused | no extra fit per call |
| voting tie-break contradicted its own comment | `max(count, k)` | voting rule now holds on every dataset |
| sentinel values `-9999`/`9999` treated as data | undefined values are `None` and can never win | no bogus winner at k=1/2 |
| division by zero on zero-variance data | `_ratio()` returns `None` | crash → handled |
| `max_k` > `n_samples` | `effective_max_k` clamps to `n_samples` | crash → handled |
| stale loop state between the two passes | single pass over the stats table | methods 4/5 no longer depend on leftover state |
| caller's `**kwargs` mutated during fitting | kwargs copied per fit, `n_clusters` ignored | no side effects |
| `max_k` not validated | `ValueError` for non-positive/non-int | tested |
| fitting continued past distance 0 | stops early (distance cannot decrease further) | degenerate dataset: 30 fits → 1, sklearn warnings 1 → 0 |
| list-of-lists `stats` with positional indexing | `StatsRow` dataclass + `stats` attribute + `summary()` | per-method results are inspectable and testable |
| each method embedded in one big loop | five small methods | each method testable independently |
| typos / `pylint: disable=C0301` / no packaging | README typos fixed, line length 100, `pyproject.toml`, `.gitignore`, CI, type hints, docstrings | — |

## Notes kept as-is on purpose

* Method 1 mixes a derivative (distance decrease) with a plain `1/k` complexity cost
  penalty; a true derivative would be `1/k - 1/(k-1)`. Measured, the "fix" is worse: method 1
  agreement with the known k falls from 85/120 to 35/120 runs on separated datasets and the
  final vote accuracy stays at 97/120, while the derivative form degenerates into "largest
  single drop" (k = 2 in 179/190 runs). The documented formula is preserved and the naming is
  corrected in the docs. See `reports/method1_grid.txt` and `reports/method1_variants.txt`.
* Methods 2–5 report the elbow vertex (`k-1`) while method 1 reports `k`. Measured per method,
  each convention is the better one (method 1: 85/120 vs 0/120 for the opposite convention);
  forcing a uniform convention drops vote accuracy from 97/120 to 14/120, so the mixed
  convention is kept (`reports/convention.txt`) and is stated explicitly in the docstring.
* Methods 4/5 depend on how the two axes are normalised to `[0,1]`, so their answers
  depend on `max_k` (final answer changes on 2/24 separated datasets when max_k varies over
  10/20/30/40, `reports/max_k_sensitivity.txt`). Documented, behaviour unchanged.
* Instability on ambiguous data (overlapping/anisotropic/iris) is inherent to elbow
  heuristics, not a bug: the same datasets still vary across `random_state`. A quality
  metric (e.g. silhouette) should be used alongside the vote; it was not added to keep
  the vote exactly the documented five methods.
