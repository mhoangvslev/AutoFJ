import shutil
from .join_function_space.autofj_join_function_space import AutoFJJoinFunctionSpace, AutoFJJoinFunctionSpacePred
from .blocker.autofj_blocker import AutoFJBlocker
from .optimizer.autofj_multi_column_greedy_algorithm import \
    AutoFJMulticolGreedyAlgorithm
import pandas as pd
from .utils import print_log
import os
from .negative_rule import NegativeRule
import numpy as np

from sklearn.base import BaseEstimator

import multiprocessing
multiprocessing.set_start_method("fork") # Fork by default, spawn for use with GPU
class AutoFJTrainer(object):
    """
    AutoFJ automatically produces record pairs that approximately match in 
    two tables L and R. It proceeds to configure suitable parameters 
    automatically, which when used to fuzzy-join L and R, meets the 
    user-specified precision target, while maximizing recall.
    
    AutoFJ attempts to solve many-to-one join problems, where each record in R
    will be joined with at most one record in L, but each record in L can be 
    joined with multiple records in R. In AutoFJ, L refers to a reference 
    table, which is assumed to be almost "duplicate-free".

    Parameters
    ----------
    precision_target: float, default=0.9
        Precision target.

    join_function_space: string or dict or list of objects, default="autofj_sm"
        There are following three ways to define the space of join functions:
        (1) Use the name of built-in join function space. There are three
        options, including "autofj_lg", "autofj_lg" and "autofj_sm" that use
        136, 68 and 14 join functions, respectively. Using less join functions
        can improve efficiency but may worsen performance.
        (2) Use a dict specifying the options for preprocessing methods,
        tokenization methods, token weighting methods and distance functions.
        The space will be the cartesian product of all options in the dict.
        See ./join_function_space/options.py for defining join functions using
        a dict.
        (3) Use a list of customized JoinFunction objects.

    distance_threshold_space: int or list, default=50
        The number of candidate distance thresholds or a list of candidate
        distance thresholds in the space.  If the number of distance thresholds
        (integer) is given, distance thresholds are spaced evenly from 0 to 1.
        Otherwise, it should be a list of floats from 0 to 1.

    column_weight_space: int or list, default=10
        The number of candidate column weights or a list of candidate
        column weights in the space. If the number of column weights
        (integer) is given, column weights are spaced evenly from 0 to 1.
        Otherwise, it should be a list of floats from 0 to 1.

    blocker: a Blocker object or None, default None
        A Blocker object that performs blocking on two tables. If None, use 
        the built-in blocker. For customized blocker, see Blocker class.

    n_jobs : int, default=-1
        Number of CPU cores used. -1 means using all processors.

    verbose: bool, default=False
        Whether to print logging
    """

    def __init__(self,
                 precision_target=0.9,
                 join_function_space="autofj_sm",
                 distance_threshold_space=50,
                 column_weight_space=10,
                 blocker=None,
                 n_jobs=-1,
                 verbose=False):
        self.precision_target = precision_target
        self.join_function_space = join_function_space

        if type(distance_threshold_space) == int:
            self.distance_threshold_space = list(
                np.linspace(0, 1, distance_threshold_space))
        else:
            self.distance_threshold_space = distance_threshold_space

        if type(column_weight_space) == int:
            self.column_weight_space = list(
                np.linspace(0, 1, column_weight_space))
        else:
            self.column_weight_space = column_weight_space

        if blocker is None:
            self.blocker = AutoFJBlocker(n_jobs=n_jobs)
        else:
            self.blocker = blocker

        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.verbose = verbose

    def join(self, left_table, right_table, id_column, on=None):
        """Join left table and right table.

        Parameters
        ----------
        left_table: pd.DataFrame
            Reference table. The left table is assumed to be almost
            duplicate-free, which means it has no or only few duplicates.

        right_table: pd.DataFrame
            Another input table.

        id_column: string
            The name of id column in the two tables. This column will not be
            used to join two tables.

        on: list or None
            A list of column names (multi-column fuzzy join) that the two tables
            will be joined on. If None, two tables will be joined on all columns
            that exist in both tables, excluding the id column.

        Returns:
        --------
            result: pd.DataFrame
                A table of joining pairs. The columns of left table are
                suffixed with "_l" and the columns of right table are suffixed
                with "_r"
        """
        left = left_table.copy(deep=True)
        right = right_table.copy(deep=True)

        # create internal id columns (use internal ids)
        left["autofj_id"] = range(len(left))
        right["autofj_id"] = range(len(right))

        # remove original ids
        left.drop(columns=id_column, inplace=True)
        right.drop(columns=id_column, inplace=True)

        # get names of columns to be joined
        if on is None:
            on = sorted(list(set(left.columns).intersection(right.columns)))
        elif isinstance(on, str):
            on = [on + "autofj_id"]
        elif isinstance(on, list):
            on = on + ["autofj_id"]

        left = left[on]
        right = right[on]

        # do blocking
        if self.verbose:
            print_log("Start blocking")
        LL_blocked = self.blocker.block(left, left, "autofj_id")
        LR_blocked = self.blocker.block(left, right, "autofj_id")

        # remove equi-joins on LL
        LL_blocked = LL_blocked[
            LL_blocked["autofj_id_l"] != LL_blocked["autofj_id_r"]]

        # learn and apply negative rules
        nr = NegativeRule(left, right, "autofj_id")
        nr.learn(LL_blocked)
        LR_blocked = nr.apply(LR_blocked)

        # create join function space
        jf_space = AutoFJJoinFunctionSpace(self.join_function_space,n_jobs=self.n_jobs, verbose=self.verbose)

        # compute distance
        if self.verbose:
            print_log("Start computing distances. Size of join function space: {}"
                      .format(len(jf_space.join_functions)))

        LL_distance, LR_distance = jf_space.compute_distance(left,
                                                             right,
                                                             LL_blocked,
                                                             LR_blocked)

        # run greedy algorithm
        if self.verbose:
            print_log("Start running greedy algorithm.")

        optimizer = AutoFJMulticolGreedyAlgorithm(
            LL_distance,
            LR_distance,
            precision_target=self.precision_target,
            candidate_thresholds=self.distance_threshold_space,
            candidate_column_weights=self.column_weight_space,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        self.selected_column_weights, self.selected_join_configs, LR_joins = \
            optimizer.run()

        if LR_joins is None:
            print("Warning: The precision target cannot be achieved.",
                  "Try a lower precision target or a larger space of join functions,",
                  "distance thresholds and column weights.")
            LR_joins = pd.DataFrame(columns=[c+"_l" for c in left_table.columns]+
                                            [c+"_r" for c in right_table.columns])
            return LR_joins

        # merge with original left and right tables
        left_idx = [l for l, r in LR_joins]
        right_idx = [r for l, r in LR_joins]
        L = left_table.iloc[left_idx].add_suffix("_l").reset_index(drop=True)
        R = right_table.iloc[right_idx].add_suffix("_r").reset_index(drop=True)
        result = pd.concat([L, R], axis=1).sort_values(by=id_column + "_r")
        return result

class AutoFJPredictor(object):
    def __init__(self, uoc, column_weights,blocker=None, n_jobs=-1, verbose=False):
        if blocker is None:
            self.blocker = AutoFJBlocker(n_jobs=n_jobs)
        else:
            self.blocker = blocker

        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.verbose = verbose

        self.uoc = uoc
        self.column_weights = column_weights
        self.jf_space = AutoFJJoinFunctionSpacePred(self.uoc, n_jobs=self.n_jobs, verbose=self.verbose)
        self.join_functions = list(map(lambda x: x.name, self.jf_space.join_functions))

    def keep_min_distance(self, LR_distance):
        # for each rid only keep lid with the smallest distance
        LR_small = {}
        for config, LR in LR_distance.items():
            LR_s = LR.sort_values(by=["distance", "autofj_id_l"], ascending=True).drop_duplicates(subset="autofj_id_r", keep="first")
            LR_s = LR_s.sort_values(by=["autofj_id_r", "autofj_id_l"])
            LR_small[config] = LR_s
        return LR_small

    def get_weighted_distance(self, LL_distance, column_weights):
        """LL_w: {config: [lid, rid, distance]}"""
        LL_w = {}
        for config, distance in LL_distance.items():
            columns = [c for c in distance.columns if c not in ["autofj_id_l", "autofj_id_r"]]
            weights = []
            for c in columns:
                if c in column_weights:
                    weights.append(column_weights[c])
                else:
                    weights.append(0)
            weights = np.array(weights).reshape(-1, 1)
            weighted_dist = distance[columns].values.dot(weights).ravel()
            LL_w[config] = pd.DataFrame({
                "autofj_id_l": distance["autofj_id_l"],
                "autofj_id_r": distance["autofj_id_r"],
                "distance": weighted_dist
            })
        return LL_w

    def scale_candidate_thresholds(self, unscaled_thresholds):
        """Scale candidate thresholds for each join function such that the
        threshold is in the range of [min, max] of LR distance.
        """
        candidate_thresholds = {}
        unscaled_thresholds = np.array(unscaled_thresholds)

        for jf in self.join_functions:
            max_d = self.LR_distance[jf]["distance"].values.max()
            min_d = self.LR_distance[jf]["distance"].values.min()
            cand_thresh = unscaled_thresholds * (max_d - min_d) + min_d
            candidate_thresholds[jf] = set(cand_thresh.tolist())
        return candidate_thresholds  
    
    def groupby_rid(self, LL_distance):
        """Group the distance table by rid. lids and distance for one rid will
        be put in one table.
        """
        LL_group = {}
        for config, LL in LL_distance.items():
            rids = LL["autofj_id_r"].values
            break_indices = np.argwhere(np.diff(rids) > 0).ravel()
            values = np.split(LL, break_indices + 1)
            keys = [rids[0]] + rids[break_indices + 1].tolist()
            LL_dict = {keys[i]: values[i] for i in range(len(values))}
            LL_group[config] = LL_dict
        return LL_group

    def join(self, left_table, right_table, id_column, on=None, cache_dir="autofj_temp"):

        if os.path.exists(cache_dir):
            print("Removing cache from previous runs...")
            shutil.rmtree(cache_dir)

        left = left_table.copy(deep=True)
        right = right_table.copy(deep=True)

        # create internal id columns (use internal ids)
        left["autofj_id"] = range(len(left))
        right["autofj_id"] = range(len(right))

        # remove original ids
        left.drop(columns=id_column, inplace=True)
        right.drop(columns=id_column, inplace=True)

        # get names of columns to be joined
        if on is None:
            on = sorted(list(set(left.columns).intersection(right.columns)))
        elif isinstance(on, str):
            on = [on] + ["autofj_id"]
        elif isinstance(on, list):
            on = on + ["autofj_id"]

        left = left[on]
        right = right[on]

        # do blocking
        if self.verbose:
            print_log("Start blocking")
        
        LL_blocked = self.blocker.block(left, left, "autofj_id")
        LR_blocked = self.blocker.block(left, right, "autofj_id")

        # remove equi-joins on LL
        LL_blocked = LL_blocked[LL_blocked["autofj_id_l"] != LL_blocked["autofj_id_r"]]

        # learn and apply negative rules
        nr = NegativeRule(left, right, "autofj_id")
        nr.learn(LL_blocked)
        LR_blocked = nr.apply(LR_blocked)

        # Transform Union of Configs
        self.LL_distance, self.LR_distance = self.jf_space.compute_distance(left, right, LL_blocked, LR_blocked)
        self.LL_distance = self.get_weighted_distance(self.LL_distance, self.column_weights)
        self.LR_distance = self.get_weighted_distance(self.LR_distance, self.column_weights)
        self.join_functions = self.LR_distance.keys()
        
        self.LL_distance = self.groupby_rid(self.LL_distance)
        self.LR_distance = self.keep_min_distance(self.LR_distance)
        
        left_idx = []
        right_idx = []

        # Where the magic happens
        uoc_df = pd.DataFrame(self.uoc, columns=["join_function", "threshold"])
        for jf, distance_df in self.LR_distance.items():
            threshold = uoc_df[uoc_df["join_function"] == jf]["threshold"].values
            valid = distance_df["distance"].apply(lambda x: any(x <= threshold))
            LR_joins = distance_df[valid]
            left_idx.extend(LR_joins["autofj_id_l"].values)
            right_idx.extend(LR_joins["autofj_id_r"].values)

        # merge with original left and right tables
        L = left_table.iloc[left_idx].add_suffix("_l").reset_index(drop=True)
        R = right_table.iloc[right_idx].add_suffix("_r").reset_index(drop=True)
        result = pd.concat([L, R], axis=1).drop_duplicates().sort_values(by=id_column + "_r")

        return result    

class AutoFJ(BaseEstimator):
    def __init__(self,
                 precision_target=0.9,
                 join_function_space="autofj_sm",
                 distance_threshold_space=50,
                 column_weight_space=10,
                 blocker=None,
                 n_jobs=-1,
                 verbose=False):
        self.trainer = AutoFJTrainer(precision_target, join_function_space, distance_threshold_space, column_weight_space, blocker, n_jobs, verbose)
        self.blocker = blocker
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.selected_column_weights = None
        self.selected_join_config = None

    def fit(self, X, y=None, **kwargs):
        left, right = X
        self.train_results = self.trainer.join(left, right, kwargs.get('id_column'), kwargs.get('on'))
        self.selected_join_config = self.trainer.selected_join_configs
        self.selected_column_weights = self.trainer.selected_column_weights
        return self

    def predict(self, X, **kwargs):
        predictor = AutoFJPredictor(self.selected_join_config, self.selected_column_weights, self.blocker, self.n_jobs, self.verbose)
        left, right = X
        res = predictor.join(left, right, kwargs.get('id_column'), kwargs.get('on'))
        return res

    def evalutate(self, y, y_pred, **kwargs):
        gt_joins = y[["id_l", "id_r"]].values
        pred_joins = y_pred[["id_l", "id_r"]].values

        pred_set = {tuple(sorted(j)) for j in pred_joins}
        gt_set = {tuple(sorted(j)) for j in gt_joins}

        # TP: When the prediction is in ground truth
        tp = pred_set.intersection(gt_set)

        # FP: In prediction but not in the ground truth
        fp = pred_set.difference(gt_set)

        # TN: neither in prediction or ground truth
        tn = pred_set.difference(gt_set)

        # FN: in ground truth but not in prediction
        fn = gt_set.difference(pred_set)

        precision = len(tp) / len(pred_set)
        recall = len(tp) / len(gt_set)

        f_coef = kwargs.get('f_coef')
        f_coef = 2 if f_coef is None else f_coef
        fscore = f_coef * precision * recall / (precision + recall)
        return {'precision': precision, 'recall': recall, 'f{f_coef}-score': fscore}