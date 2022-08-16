import pandas as pd
import math
import re
import numpy as np

from multiprocessing import Pool
from functools import partial

from qwikidata.sparql import return_sparql_query_results
from nltk.stem.porter import PorterStemmer

from .autofj_blocker import AutoFJBlocker


class WikidataBlocker(AutoFJBlocker):
    def __init__(self, wd_instance, num_candidates=None, n_jobs=-1):
        super().__init__(num_candidates, n_jobs)
        self.wd_instance = wd_instance

    def _preprocess(self, df, id_column):
        """ Preprocess the records: (1) get title only. (2) lowercase,
        remove punctuation and do stemming

        Parameters
        ----------
        df: pd.DataFrame
            Original table

        id_column: string
            The name of id column in two tables.

        Reutrn
        ------
        result: pd.DataFrame
            Preprocessed table that has two columns, am id column named as "id"
            and a column for preprocessed record named "value"
        """
        # get column names except id
        ps = PorterStemmer()

        # concat all columns, lowercase, remove punctuation, split by space,
        # and do stemming
        new_value = []
        for x in df['title'].values:
            concat_x = " ".join([str(i) for i in x])
            lower_x = re.sub('[^\w\s]', " ", concat_x.lower())
            stem_x = [ps.stem(w) for w in lower_x.split()]
            new_x = " ".join(stem_x)
            new_value.append(new_x)

        id_df = df[id_column].values
        result = pd.DataFrame({"id":id_df, "value":new_value})
        return result

    def block(self, left_table, right_table, id_column):
        self.id_column = id_column

        # preprocess records
        left = self._preprocess(left_table, id_column)
        right = self._preprocess(right_table, id_column)

        # get num candidates
        if self.num_candidates is None:
            self.num_candidates = min(int(math.pow(len(left), 0.5)), 50)

        # get candidates for each right record
        result = self._get_candidates_multi(right)

        result = result.rename(columns={"id_l": id_column+"_l",
                                        "id_r": id_column + "_r"})
        return result
    
    def _get_candidates_multi(self, right):
        """ Get candidates for one record in right table using multiple cpus
        Parameters:
        -----------
        right: pd.DataFrame
            Right table

        token_id_map: dict {token: left_id_set]}
            A dictionary that maps each token to id of left records contain
            the token

        token_idf_map: dict {token: idf_score}
            A dictionary that maps each token to its idf score

        token_tf_map: dict {token: {left_id: tf_score}}
            A dictionary that maps each token to its tf scores in different
            left records.

        Return:
        -------
        result: pd.DataFrame
            A table with two columns "id_l", "id_r" that are ids of candidate
            record pairs
        """
        right_groups = np.array_split(right, self.n_jobs)
        func = partial(self._get_candidates)

        with Pool(self.n_jobs) as pool:
            results = pool.map(func, right_groups)

        results = pd.concat(results, axis=0).sort_values(by=["id_r", "id_l"])
        return  results

    def _get_candidates(self, right):
        result = []
        for rid, rvalue in right[["id", "value"]].values:
            candidate_lids = self._wd_query(rvalue)
            for lid in candidate_lids:
                result.append([lid, rid])
        
        result = pd.DataFrame(result, columns=["id_l", "id_r"])
        return result
    
    def _wd_query(self, keyword):
        sparql_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX bds: <http://www.bigdata.com/rdf/search#>


        SELECT DISTINCT ?entId ?num WHERE {{
        
            SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:endpoint "www.wikidata.org";
                                wikibase:limit {self.num_candidates} ;
                                wikibase:api "EntitySearch";
                                mwapi:search "{keyword}";
                                mwapi:language "en".
                ?item wikibase:apiOutputItem mwapi:item.
                ?num wikibase:apiOrdinal true.
            }}
        
            ?entId  wdt:P31 .
            #?entId rdfs:label ?title
            #FILTER(LANG(?title) = "en")
        }}
        """

        res = return_sparql_query_results(sparql_query)
        return [ re.search("/(Q\d+)$", b['entId']['value']).group(1) for b in res["results"]["bindings"]]
