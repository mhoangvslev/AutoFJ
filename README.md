# AutoFJ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YF5luIGORFFkIWBbec9cB3XxFZaYITDd)


The official code for our SIGMOD 2021 paper: [Auto-FuzzyJoin: Auto-Program Fuzzy Similarity Joins Without Labeled Examples](https://arxiv.org/abs/2103.04489). To reproduce the main results in our paper, switch to `reproduce` branch.

AutoFJ automatically produces record pairs that approximately match in two input
tables without requiring explicit human input such as labeled training data. Using AutoFJ, 
users only need to provide two input tables, and a desired precision target (say 0.9). 
AutoFJ leverages the fact that one of the input is a reference table to 
automatically program fuzzy-joins that meet the precision target in expectation, 
while maximizing fuzzy-join recall (defined as the number of correctly joined records).

In AutoFJ, the left table refers to a reference table, which is assumed to be almost "duplicate-free". AutoFJ attempts to solve many-to-one join problems, where each record in the right table will be joined with at most one record in the left table, but each record in left table can be joined with multiple records in the right table. 

AutoFJ also provides a benchmark that contains [50 diverse datasets](https://github.com/chu-data-lab/AutomaticFuzzyJoin/blob/master/src/autofj/50-single-column-datasets.md) for single-column fuzzy-join tasks constructed from [DBPedia](https://www.dbpedia.org). 

## Installation

Install the package using pip

```
pip install autofj
```

## Usage

Let `left_table` be the reference table and `right_table` be another input table. The two tables are assumed to have the same schema and have an id column named `id_column`. To join `left_table` and `right_table` with
precision target 0.9, run the following code. The result will be a joined table of record pairs that are identified as matches from two input tables.
```python
from autofj import AutoFJ
fj = AutoFJ(precision_target=0.9)
result = fj.join(left_table, right_table, id_column)
```

To load a benchmark dataset named as `dataset_name`, run the following code. Each dataset contains a left table (reference table), a right table and a ground-truth table of matched record pairs. The id column of each dataset is named as "id" and the column to be joined is named as "title". The names of all benchmark datasets are listed [here](https://github.com/chu-data-lab/AutomaticFuzzyJoin/blob/master/src/autofj/50-single-column-datasets.md).
```python
from autofj.datasets import load_data
left_table, right_table, gt_table = load_data(dataset_name)
```
## Example
Run the following code to join the left and right table of TennisTournament dataset.
```python
from autofj.datasets import load_data
from autofj import AutoFJ, cross_validate, train_test_split
left_table, right_table, gt_table = load_data

X = (left_table, right_table)
y = gt_table

train, test = train_test_split(X, y, train_size=0.75 shuffle=True, stable_left=True)

X_train, y_train = train
X_test, y_test = test

model = AutoFJ(precision_target=0.9, verbose=False, name="Country")

# Train
model.fit(X_train, y_train, id_column="id", on=["title"])
print(model.evaluate(y_train, model.train_results_))

# Predict
y_pred = model.predict(X_test, id_column="id", on=["title"])
print(model.evaluate(y_test, y_pred))

# KFold Cross validation
results_cv = cross_validate(model, X, y, id_column="id", on=["title"], cv=5, scorer=model.evaluate, stable_left=True)
print(results_cv)
```

## Automatic workflow

The following `Snakemake` workflow will perform all the benchmarks on all datasets. For more information, refer to `Snakefile`

```bash
snakemake --cores 1 -C resultDir=path/to/benchmark dataDir=src/autofj/benchmark
```

## Documentation
```python
class AutoFJ(object):
    def __init__(self,
                 precision_target=0.9,
                 join_function_space="autofj_sm",
                 distance_threshold_space=50,
                 column_weight_space=10,
                 blocker=None,
                 n_jobs=-1,
                 verbose=False):
```

### Parameters
* **precision_target: *float*, default=0.9**<br />
    Precision target. The value is taken from 0-1. The default value is 0.9.

* **join_function_space: *string, dict or list of objects*, default="autofj_sm"**<br />
    Space of join functions. There are three ways to define the space of join functions:
    1. Use the name (string) of built-in join function space. There are three
    options, including "autofj_lg", "autofj_md" and "autofj_sm" that use
    136, 68 and 14 join functions, respectively. Using less join functions
    can improve efficiency but may worsen performance.
    2. Use a dict specifying the options for preprocessing methods,
    tokenization methods, token weighting methods and distance functions.
    The space will be the cartesian product of all options in the dict.
    See [options.py](https://github.com/chu-data-lab/AutomaticFuzzyJoin/blob/master/src/autofj/join_function_space/options.py) for defining join functions using
    a dict.
    3. Use a list of customized JoinFunction objects. Define JoinFunction class using prototype in [join_function.py](https://github.com/chu-data-lab/AutomaticFuzzyJoin/blob/master/src/autofj/join_function_space/join_function/join_function.py).

* **distance_threshold_space: *int or list of floats*, default=50**<br />
    The number of candidate distance thresholds or a list of candidate
    distance thresholds in the space.  If the number of distance thresholds
    (integer) is given, distance thresholds are spaced evenly from 0 to 1.
    Otherwise, it should be a list of floats from 0 to 1. Using fewer candidates
    can improve efficiency but may worsen performance.

* **column_weight_space: *int or list of floats*, default=10**<br />
    The number of candidate column weights or a list of candidate
    column weights in the space. If the number of column weights
    (integer) is given, column weights are spaced evenly from 0 to 1.
    Otherwise, it should be a list of floats from 0 to 1. Using fewer candidates
    can improve efficiency but may worsen performance.


* **blocker: *None or a Blocker object*, default None**<br />
    A Blocker object that performs blocking on two tables. If None, use 
    the built-in blocker. For using customized blocker, define Blocker class using prototype in [blocker.py](https://github.com/chu-data-lab/AutomaticFuzzyJoin/blob/master/src/autofj/blocker/blocker.py).

* **n_jobs : *int*, default=-1**<br />
    Number of CPU cores used. -1 means using all processors.

* **verbose: *bool*, default=False**<br />
    Whether to print logging

### Attributes
* **selected_column_weights: *dict***<br />
    The columns and column weights selected by the algorithm. The key is the 
    column name, the value is the weight selected for the column.

* **selected_join_configs: *list of tuples***<br />
    The union of join configurations selected by the algorithm. Each tuple
    (join_function, threshold) in the list is a join configuration that 
    consists of the name of the join function and its distance threshold.
  
### Methods
```python
join(left_table, right_table, id_column, on=None) 
```

Join left table and right table.

#### Parameters
* **left_table: *pandas.DataFrame***<br />
    Reference table. The left table is assumed to be almost duplicate-free, which means it has no or only few duplicates.

* **right_table: *pandas.DataFrame***<br />
    Another input table.

* **id_column: *string***<br />
    The name of id column in the two tables. This column will not be 
    used to join two tables.

* **on: *list or None*, default=None**<br />
    A list of column names (multi-column fuzzy join) that the two tables
    will be joined on. If None, two tables will be joined on all columns
    that exist in both tables, excluding the id column.
  
#### Return
* ***pandas.DataFrame***<br />
    A table of joining pairs. The columns of left table are
    suffixed with "_l" and the columns of right table are suffixed
    with "_r".