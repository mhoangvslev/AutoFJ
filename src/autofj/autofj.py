from collections import defaultdict
import imp
from itertools import product
import numbers
import pickle
import shutil
from statistics import mode
from tempfile import TemporaryDirectory
from time import time, time_ns
from turtle import right
from typing import Generator, List, Optional, Tuple, Union
from joblib import Parallel, delayed
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from sklearn import clone
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
import sklearn.model_selection as sklearn_ms
from sklearn.utils import shuffle as shuffle_data
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection._validation import _warn_or_raise_about_fit_failures, _fit_and_score, _insert_error_scores

from .join_function_space.autofj_join_function_space import AutoFJJoinFunctionSpace, AutoFJJoinFunctionSpacePred
from .blocker.autofj_blocker import AutoFJBlocker
from .optimizer.autofj_multi_column_greedy_algorithm import \
    AutoFJMulticolGreedyAlgorithm
from pandas.util import hash_pandas_object
from .utils import print_log
from .negative_rule import NegativeRule

class AutoFJAbstract(object):
    def __init__(self,
                 precision_target=0.9,
                 join_function_space="autofj_sm",
                 distance_threshold_space=50,
                 column_weight_space=10,
                 blocker=None,
                 n_jobs=-1,
                 verbose=False,
                 name=None
    ):
        pass

    def join(self, 
        left_table: pd.DataFrame, 
        right_table: pd.DataFrame, 
        id_column: str, 
        on: Optional[Union[str, List[str]]]=None,
    ):
        pass

    def compute_distance(self, 
        left_table: pd.DataFrame, 
        right_table: pd.DataFrame, 
        id_column: str, 
        on: Optional[Union[str, List[str]]]=None,
    ):
        pass

    def preprocessing(self, left_table: pd.DataFrame, right_table: pd.DataFrame, **kwargs):
        # do blocking
        if self.verbose:
            print_log("Start blocking")

        LL_blocked = self.blocker.block(left_table, left_table, "autofj_id")
        LR_blocked = self.blocker.block(left_table, right_table, "autofj_id")

        # remove equi-joins on LL
        LL_blocked = LL_blocked[
            LL_blocked["autofj_id_l"] != LL_blocked["autofj_id_r"]]

        # learn and apply negative rules
        if kwargs.get("pretrained_nr") is None:
            self.nr = NegativeRule(left_table, right_table, "autofj_id")
            print("Created negative rules from scratch.")
        else:
            self.nr = NegativeRule.load_pretrained_rules(kwargs.get("pretrained_nr"), left_table, right_table, "autofj_id")
            print(f"Loaded {len(self.nr.negative_rules)} rules from pretrained model.")
        
        # Block left_aug and itself then learn negative rules. Onced learned, proceed with original left blocked
        self.nr.learn(LL_blocked)
        LR_blocked = self.nr.apply(LR_blocked)

        return LL_blocked, LR_blocked
class AutoFJTrainer(AutoFJAbstract):
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
                 verbose=False,
                 name=None
    ):

        self.precision_target = precision_target
        self.join_function_space = join_function_space
        self.nr: NegativeRule = None

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

        self.cache_dir = TemporaryDirectory().name if name is None else f"tmp_{name}"
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.verbose = verbose

    def compute_distance(self, 
        left_table: pd.DataFrame, 
        right_table: pd.DataFrame, 
        id_column: str, 
        on: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
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
    
        left = left[on]
        right = right[on]

        LL_blocked, LR_blocked = self.preprocessing(left, right)

        # create join function space
        jf_space = AutoFJJoinFunctionSpace(self.join_function_space,n_jobs=self.n_jobs, verbose=self.verbose, cache_dir=self.cache_dir)

        # compute distance
        if self.verbose:
            print_log("Start computing distances. Size of join function space: {}"
                      .format(len(jf_space.join_functions)))

        LL_distance, LR_distance = jf_space.compute_distance(left, right, LL_blocked, LR_blocked)
        return LL_blocked, LR_blocked, LL_distance, LR_distance

    def join(self, 
        left_table: pd.DataFrame, 
        right_table: pd.DataFrame, 
        id_column: str, 
        on: Optional[Union[str, List[str]]]=None,
        **kwargs
    ):
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

        # Augment left column
        l_aug = kwargs.get("left_aug") 
        if l_aug is not None:
            print(f"Augmenting left for learning Negative Rule with {len(l_aug)} entries")
            left_table = pd.concat([left_table, l_aug]).drop_duplicates().reset_index(drop=True)

        _, _, LL_distance, LR_distance = self.compute_distance(left_table, right_table, id_column, on)

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

        self.selected_column_weights, self.selected_join_configs, LR_joins, precision_est = \
            optimizer.run()

        if LR_joins is None:
            print("Warning: The precision target cannot be achieved.",
                  "Try a lower precision target or a larger space of join functions,",
                  "distance thresholds and column weights.")
            LR_joins = pd.DataFrame(columns=[c+"_l" for c in left_table.columns]+
                                            [c+"_r" for c in right_table.columns])
            return LR_joins, precision_est

        # merge with original left and right tables
        left_idx = [l for l, r in LR_joins]
        right_idx = [r for l, r in LR_joins]
        L = left_table.iloc[left_idx].add_suffix("_l").reset_index(drop=True)
        R = right_table.iloc[right_idx].add_suffix("_r").reset_index(drop=True)
        result = pd.concat([L, R], axis=1).sort_values(by=id_column + "_r")
        return result, precision_est

class AutoFJPredictor(AutoFJAbstract):
    def __init__(self, uoc, column_weights, negative_rule, blocker=None, n_jobs=-1, verbose=False, name=None):
        if blocker is None:
            self.blocker = AutoFJBlocker(n_jobs=n_jobs)
        else:
            self.blocker = blocker

        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.verbose = verbose

        self.uoc = uoc
        self.column_weights = column_weights
        self.negative_rule = negative_rule

        self.cache_dir = TemporaryDirectory().name if name is None else f"tmp_{name}"
        self.jf_space = AutoFJJoinFunctionSpacePred(self.uoc, n_jobs=self.n_jobs, verbose=self.verbose, cache_dir=self.cache_dir)
        self.join_functions = list(map(lambda x: x.name, self.jf_space.join_functions))
        
        self.candidate_thresholds = {}
        for jf, threshold in self.uoc:
            if self.candidate_thresholds.get(jf) is None:
                self.candidate_thresholds[jf] = []
            if threshold not in self.candidate_thresholds[jf]:
                self.candidate_thresholds[jf].append(threshold)

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

    def precompute_precision(self):
        precision = {}
        for jf in self.join_functions:
            LL = self.LL_distance[jf]
            LR = self.LR_distance[jf]
            for thresh in self.candidate_thresholds[jf]:
                prec = {}

                for lid, rid, d in LR.values:
                    if d > thresh:
                        continue
                    # compute precision as 1/ #L-L joins
                    num_LL_joins = self.get_num_LL_joins(LL, lid, thresh)
                    prec[rid] = (lid, 1 / num_LL_joins)
                    #prec[rid] = (lid, num_LL_joins/len(LL))

                precision[(jf, thresh)] = prec

        return precision

    def get_num_LL_joins(self, LL, proxy, thresh):
        """ Number of L-L joins of proxy (the l record closest to each R),
        which is the number of L records that have distance smaller than
        2 * threshold to the proxy"""
        if proxy not in LL:
            return 1
        else:
            lid_df = LL[proxy]
            mask = lid_df["distance"] <= 2 * thresh
            num_proxy_joins = mask.sum() + 1
            return num_proxy_joins
    
    def update_selection(self, config):
        """Updated selected join configuration"""
        self.running_configs.append(config)

        for r, (l, p) in self.precision_cache[config].items():
            if r in self.running_l_cands:
                l_cands = self.running_l_cands[r]
                old_p = self.running_local_prec[r]

                if p > old_p:
                    self.running_local_prec[r] = p
                    self.running_LR_joins[r] = l
                    delta_TP = p - old_p
                else:
                    delta_TP = 0

                if l in l_cands:
                    delta_n_joins = 0
                else:
                    self.running_l_cands[r].add(l)
                    delta_n_joins = 1
            else:
                self.running_l_cands[r] = {l}
                self.running_local_prec[r] = p
                self.running_LR_joins[r] = l
                delta_TP = p
                delta_n_joins = 1

            self.running_TP += delta_TP
            self.running_n_joins += delta_n_joins

        self.LR_joins = [(l, r) for r, l in self.running_LR_joins.items()]

    def compute_distance(self, 
        left_table: pd.DataFrame, 
        right_table: pd.DataFrame, 
        id_column: str, 
        on: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
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

        left = left[on]
        right = right[on]

        LL_blocked, LR_blocked = self.preprocessing(left, right, pretrained_nr=self.negative_rule)

        LL_distance, LR_distance = self.jf_space.compute_distance(left, right, LL_blocked, LR_blocked)
        return LL_blocked, LR_blocked, LL_distance, LR_distance

    def join(self, 
        left_table: pd.DataFrame, 
        right_table: pd.DataFrame, 
        id_column: str, 
        on: Optional[Union[str, List[str]]]=None, 
        **kwargs
    ):
        # Augment left column
        l_aug = kwargs.get("left_aug") 
        if l_aug is not None:
            print(f"Augmenting left for learning Negative Rule with {len(l_aug)} entries")
            left_table = pd.concat([left_table, l_aug]).drop_duplicates().reset_index(drop=True)

        # Transform Union of Configs
        _, _, self.LL_distance, self.LR_distance = self.compute_distance(left_table, right_table, id_column, on)
        self.LL_distance = self.get_weighted_distance(self.LL_distance, self.column_weights)
        self.LR_distance = self.get_weighted_distance(self.LR_distance, self.column_weights)
        self.join_functions = self.LR_distance.keys()
        
        self.LL_distance = self.groupby_rid(self.LL_distance)
        self.LR_distance = self.keep_min_distance(self.LR_distance)

        self.running_l_cands = {}
        self.running_local_prec = {}
        self.running_configs = []
        self.running_LR_joins = {}
        self.running_n_joins = 0
        self.running_TP = 0

        self.LR_joins = None

        self.precision_cache = self.precompute_precision()

        for config in self.uoc:
            self.update_selection(config)

        if self.LR_joins is None:
            print("Warning: The precision target cannot be achieved.",
                  "Try a lower precision target or a larger space of join functions,",
                  "distance thresholds and column weights.")
            return pd.DataFrame(columns=[c+"_l" for c in left_table.columns]+
                                            [c+"_r" for c in right_table.columns])

        # merge with original left and right tables
        left_idx = [l for l, r in self.LR_joins]
        right_idx = [r for l, r in self.LR_joins]
        L = left_table.iloc[left_idx].add_suffix("_l").reset_index(drop=True)
        R = right_table.iloc[right_idx].add_suffix("_r").reset_index(drop=True)
        return pd.concat([L, R], axis=1).drop_duplicates().sort_values(by=id_column + "_r")

class KFold(sklearn_ms.KFold):
    """KFold implementation for AutoFJ dataset.

    Args:
        KFold (_type_): _description_
    """

    def split(self, 
        X: Tuple[pd.DataFrame, pd.DataFrame], 
        y: pd.DataFrame=None, 
        groups=None,
        stable_left=False
    ) -> Generator[Tuple[
            Tuple[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame],
            Tuple[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]
        ], None, None
    ]:
        left, right = X
        left = shuffle_data(left.copy(deep=True), random_state=self.random_state)
        right = shuffle_data(right.copy(deep=True), random_state=self.random_state)
        
        L_index_splits = np.array_split(np.arange(len(left)), self.n_splits)
        R_index_splits = np.array_split(np.arange(len(right)), self.n_splits)

        for n in range(self.n_splits):
            test_split_indexes = [ i for i in range(self.n_splits) if i != n ]
            L_test_indexes = np.concatenate(
                np.take(L_index_splits, test_split_indexes, axis=0), 
                axis=None
            )
            R_test_indexes = np.concatenate(
                np.take(R_index_splits, test_split_indexes, axis=0), 
                axis=None
            )

            if stable_left: l_train = l_test = left
            else: l_train, l_test = left.iloc[L_test_indexes], left.iloc[L_index_splits[n]]
            r_train, r_test = right.iloc[R_test_indexes], right.iloc[R_index_splits[n]]

            gt_train = y[ y['id_l'].isin(l_train['id'].values) & y['id_r'].isin(r_train['id'].values)]
            gt_test = y[ y['id_l'].isin(l_test['id'].values) & y['id_r'].isin(r_test['id'].values)]

            train = (l_train, r_train), gt_train
            test = (l_test, r_test), gt_test
            yield train, test
            
class AutoFJ(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 precision_target=0.9,
                 join_function_space="autofj_sm",
                 distance_threshold_space=50,
                 column_weight_space=10,
                 blocker=None,
                 n_jobs=-1,
                 verbose=False,
    ):
        self.precision_target = precision_target
        self.join_function_space = join_function_space
        self.distance_threshold_space = distance_threshold_space
        self.column_weight_space = column_weight_space
        self.blocker = blocker
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: Tuple[pd.DataFrame, pd.DataFrame], y: pd.DataFrame=None, **kwargs):
        left, right = X

        dataHash = (
            hash_pandas_object(left).sum() + 
            hash_pandas_object(right).sum() + 
            np.uint64(0) if y is None else hash_pandas_object(y).sum()
        )

        trainer = AutoFJTrainer(
            self.precision_target, 
            self.join_function_space, 
            self.distance_threshold_space, 
            self.column_weight_space, 
            self.blocker, 
            self.n_jobs, self.verbose,
            name=str(dataHash)
        )
  
        self.train_results_, self.precision_est_ = trainer.join(left, right, kwargs.get('id_column'), kwargs.get('on'), left_aug=kwargs.get("left_aug"))
        self.selected_join_config_ = trainer.selected_join_configs
        self.selected_column_weights_ = trainer.selected_column_weights
        self.negative_rule_ = trainer.nr
        return self

    def save_model(self, model_file: str):
        pickle.dump(self, open(model_file, "wb"))

    @staticmethod
    def load_model(model_file: str):
        autofj = pickle.load(open(model_file, "rb"))
        return autofj

    def predict(self, X, **kwargs):
        dataHash = np.sum([hash_pandas_object(df).sum() for df in X])
        predictor = AutoFJPredictor(
            self.selected_join_config_, 
            self.selected_column_weights_, 
            self.negative_rule_,
            self.blocker, 
            self.n_jobs, 
            self.verbose,
            name=str(dataHash)
        )
        left, right = X
        return predictor.join(left, right, kwargs.get('id_column'), kwargs.get('on'), left_aug=kwargs.get("left_aug"))

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame, **kwargs):
        gt_joins = y_true.astype(str)[["id_l", "id_r"]].values
        pred_joins = y_pred.astype(str)[["id_l", "id_r"]].values

        #pred_set = {tuple(sorted(j)) for j in pred_joins}
        #gt_set = {tuple(sorted(j)) for j in gt_joins}

        pred_set = {(l, r) for l, r in pred_joins}
        gt_set = {(l, r) for l, r in gt_joins}

        # TP: When the prediction is in ground truth
        tp = pred_set.intersection(gt_set)
        fp = pred_set.difference(tp)
        fn = gt_set.difference(tp)

        try: precision = len(tp) / len(pred_set)
        except: precision = np.nan

        try: recall = len(tp) / len(gt_set)
        except: recall = np.nan

        try:
            f_coef = kwargs.get('f_coef')
            f_coef = 1 if f_coef is None else f_coef
            fscore = (f_coef + 1) * precision * recall / (precision + recall)
        except:
            fscore = np.nan
        test_results = {'precision': precision, 'recall': recall, f'f{f_coef}-score': fscore}

        if kwargs.get("verbose"):
            return test_results, tp, fp, fn
        return test_results

    def featurize(self, X, id_column):
        trainer = AutoFJTrainer(
            self.precision_target, 
            self.join_function_space, 
            self.distance_threshold_space, 
            self.column_weight_space, 
            self.blocker, 
            self.n_jobs, self.verbose,
            name=None
        )

        left, right = X
        return trainer.compute_distance(left, right, id_column=id_column)

    def get_scorers(self):
        return {
            'precision_recall_fscore': lambda y_true, y_pred, **kwargs: tuple(self.evaluate(y_true, y_pred, **kwargs).values()),
            'precision': lambda y_true, y_pred, **kwargs: self.evaluate(y_true, y_pred, **kwargs)['precision'],
            'recall': lambda y_true, y_pred, **kwargs: self.evaluate(y_true, y_pred, **kwargs)['recall'],
            'fscore': lambda y_true, y_pred, **kwargs: self.evaluate(y_true, y_pred, **kwargs)['precision'],
        }

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None, stable_left=False):
    (left, right), gt = arrays
    if shuffle:
        left = shuffle_data(left.copy(deep=True), random_state=random_state)
        right = shuffle_data(right.copy(deep=True), random_state=random_state)
    
    if test_size is None and train_size is None:
        test_size = 0.25

    if train_size is None:
        train_size = 1 - test_size
    elif test_size is None:
        test_size = 1 - train_size

    if train_size == 1:
        print("WARNING: when train-size is 100%, train set and test set are identical...")
        train = test = (left, right), gt
        return train, test

    l_split_index = int(train_size * len(left))
    r_split_index = int(train_size * len(right))

    if stable_left: l_train = l_test = left
    else: l_train, l_test = left.iloc[:l_split_index], left.iloc[l_split_index:]

    r_train, r_test = right.iloc[:r_split_index], right.iloc[r_split_index:]

    gt_train = gt[ gt['id_l'].isin(l_train['id'].values) & gt['id_r'].isin(r_train['id'].values)]
    gt_test = gt[ gt['id_l'].isin(l_test['id'].values) & gt['id_r'].isin(r_test['id'].values)]

    train = (l_train, r_train), gt_train
    test = (l_test, r_test), gt_test

    return train, test

class GridSearchCV(sklearn_ms.GridSearchCV):
    def fit(self, X, y=None, *, groups=None, **fit_params):
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        if not isinstance(self.cv, numbers.Integral):
            raise ValueError('the parameter cv should be a number!')
            
        cv_orig = KFold(n_splits=self.cv)
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
