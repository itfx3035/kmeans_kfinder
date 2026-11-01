'''KMeans k finder.

Finds the best k (number of clusters) for a dataset by fitting scikit-learn
KMeans for every k in [1, max_k] and applying five heuristics to the resulting
distance curve. The final answer is the majority vote of the five methods.

All five methods work on the same table (`stats`), so the answer can always be
inspected method by method.

Each method reports k using the convention described in its own docstring: method 1
reports the winning row itself, methods 2..5 report the elbow vertex (k - 1).
'''

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import atan2, degrees, sqrt
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

# Method 2 counts only drops that are at least this share of the largest drop.
METHOD2_DROP_SHARE = 0.25


def calculate_angle(pt_a_axis_1, pt_a_axis_2, pt_b_axis_1, pt_b_axis_2,
                    pt_c_axis_1, pt_c_axis_2) -> float:
    '''Angle abc in degrees, measured at the intersection point b.

    Parameters
    ----------
    pt_a_axis_1, pt_a_axis_2:
        coordinates of the upper point
    pt_b_axis_1, pt_b_axis_2:
        coordinates of the intersection point (the vertex)
    pt_c_axis_1, pt_c_axis_2:
        coordinates of the lower point

    Returns
    -------
    float: angle of abc in [0, 360)
    '''
    angle = degrees(
        atan2(pt_c_axis_2 - pt_b_axis_2, pt_c_axis_1 - pt_b_axis_1)
        - atan2(pt_a_axis_2 - pt_b_axis_2, pt_a_axis_1 - pt_b_axis_1)
    )
    return angle + 360 if angle < 0 else angle


@dataclass(frozen=True)
class StatsRow:
    '''One row of the distance table for a single value of k.

    Every derived value is None when it is not defined for that k (first rows of
    the table, or division by zero on degenerate data). Undefined values never
    win a method.

    `complexity_penalty` is the fixed cost `1/k` of using k clusters. It is not the
    derivative of complexity; the derivative would be `1/k - 1/(k-1)`.
    '''

    k: int
    distance: float
    delta_distance_pct: Optional[float] = None
    diff_delta_distance_pct: Optional[float] = None
    delta_of_delta_distance_pct: Optional[float] = None
    complexity_penalty: Optional[float] = None
    diff_delta_distance_complexity: Optional[float] = None


class KMeansKFinder:
    '''Finds the best k for a dataset using five heuristics plus majority voting.

    Parameters
    ----------
    X:
        array-like data to fit KMeans on (same input as scikit-learn KMeans.fit).
    max_k:
        maximum k to check, 30 by default. Large max_k means many KMeans fits,
        so it can be slow. max_k is automatically limited to the number of rows
        in X, because KMeans cannot create more clusters than samples.
    **kwargs:
        named parameters passed to every KMeans instance. Set `random_state` for
        reproducible results. A `n_clusters` value here is ignored (it is set per k).

    Attributes
    ----------
    stats:
        list of StatsRow, one row per k. Populated by find_best_k().
    best_k_opts:
        dict with one entry per method (best_k_opt1..best_k_opt5). Method results
        that are undefined are stored as None. Populated by find_best_k().
    best_k:
        majority vote of the five methods, largest k wins ties. None until
        find_best_k() has been executed and no method produced a result.

    Methods
    ----------
    find_best_k():
        fit KMeans for k = 1..max_k, fill stats, run the five methods, vote.
    fit_best():
        return a fitted KMeans instance using best_k (calls find_best_k() if needed).

    Notes
    ----------
    Reporting conventions (kept as documented in the README):

    * method 1 reports the k of the row where the distance improvement minus the
      complexity cost penalty `1/k` is largest. The `1/k` term is a fixed cost, not a
      derivative - see the method docstring for why it is kept.
    * methods 2..5 report the elbow vertex, i.e. k - 1.

    Measured on 120 runs over datasets with separated clusters (see
    reports/method1_grid.txt and reports/convention.txt): using the true derivative
    `1/k - 1/(k-1)` lowers method 1 agreement with the known k from 85/120 to 35/120,
    because `1/(k(k-1))` decays as `1/k^2` and the score degenerates into "largest
    single drop", which is almost always k = 2. The final vote accuracy does not
    improve, so the documented formula is kept.
    '''

    def __init__(self, X, max_k: int = 30, **kwargs):
        if not isinstance(max_k, int) or max_k < 1:
            raise ValueError(f'max_k must be a positive integer, got {max_k!r}')

        self.max_k = max_k
        # n_clusters is set per k, so a user supplied value is ignored
        self.kmeans_kwargs = {k: v for k, v in kwargs.items() if k != 'n_clusters'}
        self.X = X

        self.stats: list[StatsRow] = []
        self.best_k_opts: dict[str, Optional[int]] = {}
        self.best_k: Optional[int] = None
        self._best_model = None

    @property
    def n_samples(self) -> int:
        return np.asarray(self.X).shape[0]

    @property
    def effective_max_k(self) -> int:
        '''max_k limited by the number of samples (KMeans cannot exceed them).'''
        return min(self.max_k, self.n_samples)

    def _fit_stats(self) -> list[StatsRow]:
        '''Fit KMeans for every k and build the stats table in one pass.

        Fitting stops early when the distance reaches 0, because it cannot decrease
        further: every later k would produce the same zero distance (and a
        convergence warning from scikit-learn) without adding any information.
        '''
        rows: list[StatsRow] = []
        for k in range(1, self.effective_max_k + 1):
            km_obj = KMeans(n_clusters=k, **self.kmeans_kwargs)
            km_obj.fit(self.X)
            distance = sqrt(km_obj.inertia_)

            prev = rows[-1] if rows else None
            prev_delta = prev.delta_distance_pct if prev else None

            delta = _ratio(prev.distance - distance, prev.distance) if prev else None
            diff_delta = (delta - prev_delta) if prev and prev_delta is not None else None
            delta_of_delta = _ratio(prev_delta - delta, prev_delta) if prev and prev_delta else None
            complexity_penalty = 1 / k if prev else None
            complexity_diff = (delta - complexity_penalty) if prev and delta is not None else None

            rows.append(StatsRow(k=k, distance=distance, delta_distance_pct=delta,
                                 diff_delta_distance_pct=diff_delta,
                                 delta_of_delta_distance_pct=delta_of_delta,
                                 complexity_penalty=complexity_penalty,
                                 diff_delta_distance_complexity=complexity_diff))

            if distance == 0:
                break

        self.stats = rows
        return rows

    # ------------------------------------------------------------------ methods

    def _method1(self, rows) -> Optional[int]:
        '''Row with the largest distance improvement minus complexity cost penalty.

        Score: `delta_distance_pct - 1/k`. The `1/k` term is the fixed cost of using k
        clusters, not the derivative of complexity (that would be
        `1/k - 1/(k-1) = -1/(k(k-1))`). Comparing a relative improvement with a fixed
        cost is what makes this heuristic elbow-like: a true derivative decays as 1/k^2,
        so the score degenerates into "largest single drop" and nearly always returns
        k = 2 - measured 179/185 runs, with agreement with the known k falling from
        85/120 to 35/120 on separable data (reports/method1_grid.txt).

        Reported value: the k of the winning row itself. Methods 2..5 report the elbow
        vertex (k - 1); each convention is the one that fits what that method measures
        (reports/convention.txt).
        '''
        best_k, best_value = None, None
        for row in rows:
            if row.k < 2 or row.diff_delta_distance_complexity is None:
                continue
            if best_value is None or row.diff_delta_distance_complexity > best_value:
                best_value = row.diff_delta_distance_complexity
                best_k = row.k
        return best_k

    def _method2(self, rows) -> Optional[int]:
        '''Previous k of the last significant drop of the distance derivative.

        Only drops that are at least METHOD2_DROP_SHARE of the largest drop count.
        Reported value: the vertex k - 1.
        '''
        drops = [r.diff_delta_distance_pct for r in rows
                 if r.diff_delta_distance_pct is not None]
        if not drops:
            return None

        largest_drop = min(drops)          # most negative value
        threshold = largest_drop * METHOD2_DROP_SHARE

        best_k = None
        for row in rows:
            if row.k < 2 or row.diff_delta_distance_pct is None:
                continue
            if row.diff_delta_distance_pct < threshold:
                best_k = row.k - 1
        return best_k

    def _method3(self, rows) -> Optional[int]:
        '''Elbow vertex of the second order derivative of the distance.

        Reported value: the elbow vertex k - 1.
        '''
        best_k, best_value = None, None
        for row in rows:
            if row.k < 3 or row.delta_of_delta_distance_pct is None:
                continue
            if best_value is None or row.delta_of_delta_distance_pct > best_value:
                best_value = row.delta_of_delta_distance_pct
                best_k = row.k - 1
        return best_k

    def _method4(self, rows) -> Optional[int]:
        '''Elbow vertex of the triangle built from three neighbouring points.

        Reported value: the elbow vertex k - 1.
        '''
        scaled = self._scaled(rows)
        if scaled is None:
            return None
        best_k, best_height = None, None
        for i in range(2, len(rows)):
            p1, p2, p3 = scaled[i - 2], scaled[i - 1], scaled[i]
            a_side = _side(p1, p2)
            b_side = _side(p2, p3)
            c_side = _side(p1, p3)
            if not c_side:
                continue

            half_p = (a_side + b_side + c_side) / 2
            area_sq = half_p * (half_p - a_side) * (half_p - b_side) * (half_p - c_side)
            if area_sq < 0:
                continue
            height = 2 * sqrt(area_sq) / c_side

            if best_height is None or height > best_height:
                best_height = height
                best_k = rows[i].k - 1
        return best_k

    def _method5(self, rows) -> Optional[int]:
        '''True elbow: minimal angle on the distance plot.

        Reported value: the elbow vertex k - 1.
        '''
        scaled = self._scaled(rows)
        if scaled is None:
            return None
        best_k, best_angle = None, None
        for i in range(2, len(rows)):
            p1, p2, p3 = scaled[i - 2], scaled[i - 1], scaled[i]
            angle = calculate_angle(p3[0], p3[1], p2[0], p2[1], p1[0], p1[1])
            if best_angle is None or angle < best_angle:
                best_angle = angle
                best_k = rows[i].k - 1
        return best_k

    def _scaled(self, rows):
        '''Points of the plot normalised to a [0, 1] x [0, 1] box.

        Returns None for degenerate data (zero total distance), where the plot has
        no shape and methods 4 and 5 are not defined.

        The x axis is scaled by effective_max_k, so methods 4 and 5 can change their
        answer when max_k changes. That is the documented behaviour.
        '''
        max_distance = max(row.distance for row in rows)
        if not max_distance:
            return None
        limit = self.effective_max_k
        return [(row.k / limit, row.distance / max_distance) for row in rows]

    # ------------------------------------------------------------------ voting

    @staticmethod
    def _vote(best_k_opts) -> Optional[int]:
        '''Majority vote of the method results, largest k wins ties.'''
        counts = Counter(v for v in best_k_opts.values() if v is not None)
        if not counts:
            return None
        top = max(counts.values())
        return max(k for k, count in counts.items() if count == top)

    def find_best_k(self) -> Optional[int]:
        '''Run all five methods and return the majority vote.

        Returns
        -------
        int or None: best k, or None when no method produced a result.
        '''
        rows = self._fit_stats()

        self.best_k_opts = {
            'best_k_opt1': self._method1(rows),
            'best_k_opt2': self._method2(rows),
            'best_k_opt3': self._method3(rows),
            'best_k_opt4': self._method4(rows),
            'best_k_opt5': self._method5(rows),
        }
        self.best_k = self._vote(self.best_k_opts)
        return self.best_k

    def fit_best(self):
        '''Return a fitted KMeans instance using best_k.

        find_best_k() is executed if it has not been executed yet. The fitted model
        is cached, so repeated calls do not refit.
        '''
        if self.best_k is None:
            self.find_best_k()
        if self._best_model is None:
            if self.best_k is None:
                raise ValueError('no method produced a result, cannot fit a model')
            self._best_model = KMeans(n_clusters=self.best_k, **self.kmeans_kwargs)
            self._best_model.fit(self.X)
        return self._best_model

    def summary(self) -> str:
        '''Human readable summary of the stats table and the method results.'''
        lines = [f'{"k":>3} {"distance":>10} {"delta":>8} {"d_delta":>10} '
                 f'{"d2_delta":>10} {"m1_score":>10}']
        for row in self.stats:
            lines.append(
                f'{row.k:>3} {row.distance:>10.4f} '
                f'{_fmt(row.delta_distance_pct):>8} {_fmt(row.diff_delta_distance_pct):>10} '
                f'{_fmt(row.delta_of_delta_distance_pct):>10} '
                f'{_fmt(row.diff_delta_distance_complexity):>10}'
            )
        lines.append(f'methods: {self.best_k_opts}')
        lines.append(f'best_k: {self.best_k}')
        return '\n'.join(lines)


def _ratio(numerator, denominator):
    '''Division that returns None instead of raising on zero denominators.'''
    if not denominator:
        return None
    return numerator / denominator


def _side(p1, p2):
    return sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)


def _fmt(value):
    return 'n/a' if value is None else f'{value:.4f}'
