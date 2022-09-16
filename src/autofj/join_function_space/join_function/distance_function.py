"""Compute distance"""
import editdistance
import jellyfish
import collections
from collections import Counter
import time
import numpy as np
import pandas as pd
import spacy
from transformers import BartTokenizer, BartModel
from scipy.spatial.distance import euclidean, cosine
import torch
from dateutil.parser import parse as date_parse

# global tokenizer
# global model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# model = BartModel.from_pretrained("facebook/bart-base").to(device)

#import spacy_universal_sentence_encoder

"""Distance Functions"""
def jaccardDistance(x, y, w=None):
    inter = set(x).intersection(set(y))
    union = set(x).union(set(y))
    if w is None:
        sum_inter = len(inter)
        sum_union = len(union)
    else:
        sum_inter = sum([w[s] for s in inter])
        sum_union = sum([w[s] for s in union])
    d = 1 - sum_inter / (sum_union + 1e-9)
    return d

def cosineDistance(x, y, w=None):
    c1 = Counter(x)
    c2 = Counter(y)
    inter = set(x).intersection(set(y))

    if w is None:
        uv = sum([c1[s]*c2[s] for s in inter])
        u = np.sqrt(sum([c1[s]**2 for s in set(x)]))
        v = np.sqrt(sum([c2[s]**2 for s in set(y)]))
    else:
        uv = sum([w[s]*c1[s]*w[s]*c2[s] for s in inter])
        u = np.sqrt(sum([(w[s]*c1[s])**2 for s in set(x)]))
        v = np.sqrt(sum([(w[s]*c2[s])**2 for s in set(y)]))

    d = 1 - uv / (u * v + 1e-9)
    return d

def diceDistance(x, y, w=None):
    inter = set(x).intersection(set(y))
    union = set(x).union(set(y))
    if w is None:
        sum_inter = len(inter)
        sum_union = len(union)
    else:
        sum_inter = sum([w[s] for s in inter])
        sum_union = sum([w[s] for s in union])
    d = 1 - (2 * sum_inter / (sum_inter + sum_union + 1e-9))
    return d

def maxincDistance(x, y, w=None):
    inter = set(x).intersection(set(y))
    if w is None:
        sum_inter = len(inter)
    else:
        sum_inter = sum([w[s] for s in inter])

    if w is None:
        sum_x = len(set(x))
        sum_y = len(set(y))
    else:
        sum_x = sum([w[s] for s in set(x)])
        sum_y = sum([w[s] for s in set(y)])
    min_sum = min(sum_x, sum_y)
    d = 1 - (sum_inter / (min_sum + 1e-9))
    return d

def intersectDistance(x, y, w=None):
    inter = set(x).intersection(set(y))
    union = set(x).union(set(y))
    if w is None:
        sum_inter = len(inter)
        sum_union = len(union)
    else:
        sum_inter = sum([w[s] for s in inter])
        sum_union = sum([w[s] for s in union])
    d = 1 - sum_inter / (sum_inter + sum_union + 1e-9)
    return d

def isContain(x, y):
    set_x = set(x)
    set_y = set(y)

    if len(set_x) > len(set_y):
        return set_y.issubset(set_x)
    else:
        return set_x.issubset(set_y)

def containCosineDistance(x, y, w=None):
    if isContain(x, y):
        return cosineDistance(x, y, w)
    else:
        return 1

def containJaccardDistance(x, y, w=None):
    if isContain(x, y):
        return jaccardDistance(x, y, w)
    else:
        return 1

def containDiceDistance(x, y, w=None):
    if isContain(x, y):
        return diceDistance(x, y, w)
    else:
        return 1

def editDistance(x, y):
    d = editdistance.eval(x, y)
    return d

def jaroDistance(x, y):
    d = 1 - jellyfish.jaro_winkler_similarity(x, y)
    return d

def knowMoreDistance(x, y):
    """Similarity as defined in KnowMore paper

    Args:
        x (any): left element
        y (any): right element

    Returns:
        float: distance measurement
    """
    if type(x) == int or type(x) == bool:
        return int(x == y)
    # If datetime
    try: 
        date_x = list(date_parse(str(x)).timetuple())
        date_y = list(date_parse(str(y)).timetuple())
        return jaccardDistance(date_x, date_y)
    except:
        return cosineDistance(x, y)

def embedDistance(x, y, embedding):
    vecX = embedding(x) 
    vecY = embedding(y)

    # Special cases
    # if isinstance(embedding, spacy.language.xx.MultiLanguage):
    #     vecX = vecX.vector
    #     vecY = vecY.vector
    return vecX.similarity(vecY)

# def BARTEmbedding(x):
#     inputs = tokenizer(x, return_tensors="pt").to(device)
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)

def euclideanEmbedDistance(x, y, embedding):
    x = embedding(x)
    y = embedding(y)
    return torch.cdist(x, y, p=2.0) 

def cosineEmbedDistance(x, y, embedding):
    x = embedding(x)
    y = embedding(y)
    res = 1-torch.nn.CosineSimilarity(dim=1)(x, y)
    return res
class DistanceFunction(object):
    """Distance function

    Parameters
    ----------
    method: string
        Method of computing distance. The available methods are listed as
        follows.
        Set-based distance
            - jaccardDistance
            - cosineDistance
            - diceDistance
            - maxincDistance
            - intersectDistance
            - containCosineDistance
            - containJaccardDistance
            - containDiceDistance
        Char-based distance
            - editDistance
            - jaroDistance

    """
    def __init__(self, method):
        self.method = method
        if method == "jaccardDistance":
            self.func = jaccardDistance
        elif method == "cosineDistance":
            self.func = cosineDistance
        elif method == "diceDistance":
            self.func = diceDistance
        elif method == "maxincDistance":
            self.func = maxincDistance
        elif method == "intersectDistance":
            self.func = intersectDistance
        elif method == "editDistance":
            self.func = editDistance
        elif method == "jaroDistance":
            self.func = jaroDistance
        elif method == "knowMoreDistance":
            self.func = knowMoreDistance
        elif method == "containCosineDistance":
            self.func = containCosineDistance
        elif method == "containJaccardDistance":
            self.func = containJaccardDistance
        elif method == "containDiceDistance":
            self.func = containDiceDistance
        elif method == "embedDistance":
            self.func = embedDistance
            #self.embedding = spacy_universal_sentence_encoder.load_model('xx_use_lg')
            self.embedding = spacy.load("en_core_web_lg")

        # BERT/BART
        # elif method == "embed_BART-Large":
        #     self.func = euclideanEmbedDistance
        #     self.embedding = BARTEmbedding
        # else:
        #     raise Exception("{} is an invalid distance function"
        #                      .format(method))

    def compute_distance(self, LR, weight=None):
        """"Compute distance score between tuple pairs

        Parameters:
        ----------
        LR: pd.DataFrame
            A table of tuple pairs. The columns of left and right values are
            named as "value_l" and "value_r". For char-based distance the type
            of values are string. For set-based distance the type of values are
            token set.

        weight: dict, default=None
            Weighting schema. If none, uniform weight or no weight is used.

        Return:
        -------
        distance: pd.Series
            distance between tuple pairs for each row
        """
        
        
        # Apply maximum distances when comparing empty strings
        def calc_distance(x, y, embedding = None, weight = None):
            """Apply distance function to x and y

            Args:
                x (List[str]): list of tokens of the left string
                y (List[str]): list of tokens of the right string
                embedding (_type_, optional): _description_. Defaults to None.
                weight (_type_, optional): _description_. Defaults to None.

            Returns:
                float: distance [0, 1] between left and right 
            """
            if len(x) == 0 or len(y) == 0:
                return np.nan
            elif embedding is not None:
                return self.func(x, y, embedding)
            elif weight is not None:
                return self.func(x, y, weight)
            return self.func(x, y)

        if weight is None:
            if "embed" not in self.method:
                distance = LR.apply(lambda x: calc_distance(x.value_l, x.value_r), axis=1)
            else:
                distance = LR.apply(lambda x: calc_distance(x.value_l, x.value_r, embedding=self.embedding), axis=1)
        else:
            distance = LR.apply(lambda x: calc_distance(x.value_l, x.value_r, weight), axis=1)
        return distance.fillna(distance.max())

# data = pd.read_csv("../../data/left.csv")["title"]
# X = np.concatenate([data.values for _ in range(20)])
# X = pd.Series(X)
#
# L = X
# R = X.sample(frac=1)
#
# from tokenizer import Tokenizer
# tokenizer = Tokenizer("splitBySpace")
# L = tokenizer.tokenize(L)
# R = tokenizer.tokenize(R)
# LR = pd.DataFrame({"value_l":L, "value_r":R})
#
# tic = time.time()
# methods = ["jaccardDistance", "maxincDistance", "containCosineDistance"]
# distance_function = DistanceFunction("jaccardDistance")
# distance_function.compute_distance(LR)
# distance_function = DistanceFunction("maxincDistance")
# distance_function.compute_distance(LR)
# distance_function = DistanceFunction("containCosineDistance")
# distance_function.compute_distance(LR)
# print(time.time() - tic)
