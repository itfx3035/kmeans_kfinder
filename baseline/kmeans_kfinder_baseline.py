# pylint: disable=C0301

''' KMeans k finder
    Finds the best k (number of clusters) on supplied dataset.
    Works upon sci-kit learn KMeans implementation.
    Uses five different methods to find best k; final result is based on voting.
'''

from collections import Counter
from math import sqrt, atan2, degrees
from sklearn.cluster import KMeans

def calculate_angle(pt_a_axis_1, pt_a_axis_2, pt_b_axis_1, pt_b_axis_2, pt_c_axis_1, pt_c_axis_2):
    ''' Calculates angle between two lines

        Parameters
        ------------------
        pt_a_axis_1, pt_a_axis_2: float
            axis coordinates (x,y) of upper point
        pt_b_axis_1, pt_b_axis_2: float
            axis coordinates (x,y) of intersection point
        pt_c_axis_1, pt_c_axis_2: float
            axis coordinates (x,y) of lower point

        Returns
        ------------------
        angle of abc: float
    '''
    angle = degrees(atan2(pt_c_axis_2-pt_b_axis_2, pt_c_axis_1-pt_b_axis_1) -
                    atan2(pt_a_axis_2-pt_b_axis_2, pt_a_axis_1-pt_b_axis_1))
    return angle + 360 if angle < 0 else angle

class KMeansKFinder:
    ''' Parameters for initializing
        ----------------------------
        X - array-like or similar
            Data to fit KMeans model. Refer to sci-kit learn KMeans fit method.
        max_k - integer
            Maximum k to check. max_k=30 by default.
            Note that setting large max_k may cause performance degradation.
        **kwargs - named parameters to pass to KMeans object.

        Properties of model object
        ----------------------------
        max_k: integer
            Maximum k to check
        kmeans_kwargs - **kwargs - named parameters to pass to KMeans object
        X:  array-like or similar
            Data to fit KMeans model. Refer to sci-kit learn KMeans fit method
        best_k: integer
            best k according to majority of methods.
            find_best_k method must be executed to find the best k
        best_k_opts: dict with 5 elements
            best_k_opt1: best k according to method #1
                k at max difference between delta_kmeans_distance_pct and diff_delta_distance_complexity,
                where delta_kmeans_distance_pct represents KMeans distances decrease (derivative from kmeans distance)
                and diff_delta_distance_complexity represents inverted complexity increase (basically 1/k)
            best_k_opt2: best k according to method #2
                k, which is previous to last significant drop of delta_kmeans_distance_pct,
                where delta_kmeans_distance_pct represents KMeans distances decrease (derivative from kmeans distance)
                and only top bigger 25% of drops are taken into account
            best_k_opt3: best k according to method #3
                k at max delta_of_delta_kmeans_distance_pct,
                where delta_of_delta_kmeans_distance_pct is derivative from of delta_kmeans_distance_pct,
                which means that delta_of_delta_kmeans_distance_pct is a second order derivative from kmeans distance
            best_k_opt4: best k according to method #4
                k at max triangle height,
                where tringle consists of three nearest points of kmeans distance plot
            best_k_opt5: best k according to method #5
                k at minimal angle on kmeans distance plot. True elbow method.
    '''

    def __init__(self, X, max_k=30, **kwargs):
        self.max_k = max_k
        self.kmeans_kwargs = kwargs
        self.X = X
        best_k = None
        best_k_opts = None

    def find_best_k(self):
        ''' Find best k parameter (number of clusters)

            Results
            -----------------------
            best_k: integer
                best k according to majority of methods
        '''
        kwargs = self.kmeans_kwargs

        # results
        best_k_opts = {}

        # init variables
        kmeans_distance_k_minus_2 = 0
        kmeans_distance_k_minus_1 = 0
        max_diff_delta_distance_complexity = -9999
        max_delta_of_delta_kmeans_distance_pct = -9999
        max_drop_delta_kmeans_distance_pct = 9999
        max_triangle_height = -9999
        max_kmeans_distance = -9999
        min_angle = 360
        delta_kmeans_distance_pct_k_minus_1 = -9999

        # statistic storage
        stats = []

        for number_of_k in range (1,self.max_k+1):
            kwargs['n_clusters'] = number_of_k
            km_obj =  KMeans(**kwargs)
            km_obj.fit(self.X)
            kmeans_distance = sqrt(km_obj.inertia_)

            # calculate required values
            if number_of_k==1:
                delta_complexity_pct = 0
                delta_kmeans_distance_pct = 0
                delta_of_delta_kmeans_distance_pct = 0
            else:
                delta_complexity_pct = 1/number_of_k
                delta_kmeans_distance_pct = (kmeans_distance_k_minus_1 - kmeans_distance)/kmeans_distance_k_minus_1
                max_drop_delta_kmeans_distance_pct = min(max_drop_delta_kmeans_distance_pct, diff_delta_kmeans_distance_pct)

                if number_of_k>2:
                    delta_of_delta_kmeans_distance_pct = (delta_kmeans_distance_pct_k_minus_1 - delta_kmeans_distance_pct)/delta_kmeans_distance_pct_k_minus_1
                else:
                    delta_of_delta_kmeans_distance_pct = 0

            diff_delta_distance_complexity = delta_kmeans_distance_pct - delta_complexity_pct

            # max delta_kmeans_distance_pct drop
            max_kmeans_distance = max(max_kmeans_distance, kmeans_distance)

            diff_delta_kmeans_distance_pct = delta_kmeans_distance_pct - delta_kmeans_distance_pct_k_minus_1

            stats.append([number_of_k, kmeans_distance, delta_kmeans_distance_pct,
                          diff_delta_kmeans_distance_pct, delta_of_delta_kmeans_distance_pct,
                          diff_delta_distance_complexity])

            kmeans_distance_k_minus_2 = kmeans_distance_k_minus_1
            kmeans_distance_k_minus_1 = kmeans_distance
            delta_kmeans_distance_pct_k_minus_1 = delta_kmeans_distance_pct

        top025_drop = max_drop_delta_kmeans_distance_pct*0.25

        # apply all methods to find best k
        for stat in stats:
            number_of_k = stat[0]
            kmeans_distance = stat[1]
            delta_kmeans_distance_pct = stat[2]
            diff_delta_kmeans_distance_pct = stat[3]
            delta_of_delta_kmeans_distance_pct = stat[4]
            diff_delta_distance_complexity = stat[5]

            # method 1 - max difference between delta_kmeans_distance_pct and diff_delta_distance_complexity
            if number_of_k>1 and max_diff_delta_distance_complexity<diff_delta_distance_complexity:
                max_diff_delta_distance_complexity = diff_delta_distance_complexity
                best_k_opts['best_k_opt1'] = number_of_k

            # method 2 - prev k of last significant drop of delta_kmeans_distance_pct
            if number_of_k>1 and diff_delta_kmeans_distance_pct<top025_drop:
                best_k_opts['best_k_opt2'] = number_of_k-1

            # method 3 - max delta_of_delta_kmeans_distance_pct
            if number_of_k>1 and delta_of_delta_kmeans_distance_pct>max_delta_of_delta_kmeans_distance_pct:
                max_delta_of_delta_kmeans_distance_pct = delta_of_delta_kmeans_distance_pct
                best_k_opts['best_k_opt3'] = number_of_k-1

            # prepare data for methods 4 and 5
            scaled_kmeans_distance_k_minus_2 = kmeans_distance_k_minus_2/max_kmeans_distance
            scaled_kmeans_distance_k_minus_1 = kmeans_distance_k_minus_1/max_kmeans_distance
            scaled_kmeans_distance = kmeans_distance/max_kmeans_distance

            scaled_complexity_k_minus_2 = (number_of_k-2)/self.max_k
            scaled_complexity_k_minus_1 = (number_of_k-1)/self.max_k
            scaled_complexity = number_of_k/self.max_k

            # method 4 - triangle height
            if number_of_k>=3:
                # sides of the triangle
                a_side = sqrt((scaled_kmeans_distance_k_minus_1 - scaled_kmeans_distance_k_minus_2)**2 + (scaled_complexity_k_minus_1 - scaled_complexity_k_minus_2)**2)
                b_side = sqrt((scaled_kmeans_distance - scaled_kmeans_distance_k_minus_1)**2 + (scaled_complexity - scaled_complexity_k_minus_1)**2)
                c_side = sqrt((scaled_kmeans_distance - scaled_kmeans_distance_k_minus_2)**2 + (scaled_complexity - scaled_complexity_k_minus_2)**2)

                # half of the perimeter
                half_p = (a_side + b_side + c_side)/2

                # height of the triangle
                t_height = 2*sqrt(half_p*(half_p-a_side)*(half_p-b_side)*(half_p-c_side))/c_side

                if t_height>max_triangle_height:
                    max_triangle_height = t_height
                    best_k_opts['best_k_opt4'] = number_of_k-1

            # method 5 - true elbow
            if number_of_k>=3:
                curr_angle = calculate_angle(scaled_complexity, scaled_kmeans_distance,
                                             scaled_complexity_k_minus_1, scaled_kmeans_distance_k_minus_1,
                                             scaled_complexity_k_minus_2, scaled_kmeans_distance_k_minus_2)
                if curr_angle<min_angle:
                    min_angle = curr_angle
                    best_k_opts['best_k_opt5'] = number_of_k-1

            kmeans_distance_k_minus_2 = kmeans_distance_k_minus_1
            kmeans_distance_k_minus_1 = kmeans_distance

        self.best_k_opts = best_k_opts

        cntr = Counter(list(best_k_opts.values()))
        best_k = 0
        best_k_cnt = 0
        for element in cntr:
            if cntr[element]>=best_k_cnt: # larger k with same score is preferable
                best_k_cnt = cntr[element]
                best_k = element

        self.best_k = best_k
        return best_k


    def fit_best(self):
        ''' Fits scikit-learn Kmeans with best k found.
            find_best_k will be executed in order to calculate best k if it wasn't executed previously.

            Returns
            ------------------
            Instance of scikit-learn Kmeans, fitted with best_k
        '''
        if self.best_k is None:
            self.find_best_k()

        kwargs = self.kmeans_kwargs
        kwargs['n_clusters'] = self.best_k
        km_obj =  KMeans(**kwargs)
        km_obj.fit(self.X)

        return km_obj
