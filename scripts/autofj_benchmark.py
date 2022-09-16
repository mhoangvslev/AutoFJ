from copyreg import pickle
from email.policy import default
from glob import glob
import pathlib
import shutil
from time import time
from autofj import AutoFJ
import pandas as pd
import numpy as np
import os
import click

from autofj import train_test_split
from autofj.autofj import KFold
from tqdm import tqdm

from autofj.blocker.knowmore_blocker import KnowMoreBlocker
from autofj.blocker.autofj_blocker import AutoFJBlocker
from autofj.blocker.wikidata_blocker import WikidataBlocker

@click.group()
def cli():
    pass

@cli.command()
@click.argument("modelname", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("dataset", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--id", type=click.STRING, default="id")
@click.option("--on", type=click.STRING, default=None)
@click.option("--outdir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
def predict(modelname, dataset, id, on, outdir):
    model: AutoFJ = AutoFJ.load_model(modelname)

    data_l = pd.read_csv(os.path.join(dataset, "left.csv"))
    data_r = pd.read_csv(os.path.join(dataset, "right.csv"))
    data_gt = pd.read_csv(os.path.join(dataset, "gt.csv"))

    mergeCols = list(model.selected_column_weights_.keys())
    dropCols = [ c for c in data_l.columns if "id" not in c and c not in mergeCols ]

    if on is not None: on = on.split(",")

    #data_lt = data_l.assign(data=data_l[mergeCols].astype(str).agg(' '.join, axis=1)).drop(mergeCols, axis=1).drop(dropCols, axis=1)
    #data_rt = data_r.assign(data=data_r[mergeCols].astype(str).agg(' '.join, axis=1)).drop(mergeCols, axis=1).drop(dropCols, axis=1)

    X_test, y_test = (data_l, data_r), data_gt    
    y_pred = model.predict(X_test, id_column=id, on=on)

    test_results, tp, fp, fn = model.evaluate(y_test, y_pred, verbose=True)

    tp = (
        pd.DataFrame(tp, columns=['id_l', 'id_r'])
            .merge(data_l.rename(columns={'id': 'id_l'}), on="id_l", suffixes=("_l", "_r"))
            .merge(data_r.rename(columns={'id': 'id_r'}), on="id_r", suffixes=("_l", "_r"))
    )
    fp = (
        pd.DataFrame(fp, columns=['id_l', 'id_r'])
            .merge(data_l.rename(columns={'id': 'id_l'}), on="id_l", suffixes=("_l", "_r"))
            .merge(data_r.rename(columns={'id': 'id_r'}), on="id_r", suffixes=("_l", "_r"))
    )

    fn = (
        pd.DataFrame(fn, columns=['id_l', 'id_r'])
            .merge(data_l.rename(columns={'id': 'id_l'}), on="id_l", suffixes=("_l", "_r"))
            .merge(data_r.rename(columns={'id': 'id_r'}), on="id_r", suffixes=("_l", "_r"))
    )

    if outdir is None: outdir = os.path.dirname(modelname)

    y_pred.to_csv(os.path.join(outdir, "pred.csv"), index=False)
    tp.to_csv(os.path.join(outdir, "pred_tp.csv"), index=False)
    fp.to_csv(os.path.join(outdir, "pred_fp.csv"), index=False)
    fn.to_csv(os.path.join(outdir, "pred_fn.csv"), index=False)

    learned_col_weights = pd.DataFrame({ k: [v] for k, v in model.selected_column_weights_.items()})
    learned_col_weights.to_csv(os.path.join(outdir, "learned_col_weight.csv"), index=False)
    learned_join_conf = pd.DataFrame(model.selected_join_config_, columns=["config", "threshhold"])
    learned_join_conf.to_csv(os.path.join(outdir, "learned_join_conf.csv"), index=False)

    print(test_results)

@cli.command()
@click.argument("input-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("result-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dataset", type=click.STRING)
@click.option("--augqty", type=click.FLOAT, default=1.0)
@click.option("--verbose", type=click.BOOL, default=False)
@click.option("--jfs", type=click.STRING, default="autofj_sm")
@click.option("--name", type=click.STRING, default="default")
def train(input_dir, result_dir, dataset, augqty, verbose, jfs, name):

    tmpDirs = glob("tmp*", recursive=False)
    for tmpDir in tmpDirs: 
        shutil.rmtree(tmpDir)
        
    data_l = pd.read_csv(os.path.join(input_dir, "left.csv"))
    data_r = pd.read_csv(os.path.join(input_dir, "right.csv"))
    data_gt = pd.read_csv(os.path.join(input_dir, "gt.csv"))

    X, y = (data_l, data_r), data_gt

    model = AutoFJ(precision_target=0.9, verbose=verbose, join_function_space=jfs)

    results = {
        "train_times": [],
        "test_times": [],
        "train_scores": [],
        "test_scores": [],
        "train_precision_est": [],
        "gt_size_train": [],
        "l_size_train": [],
        "r_size_train": [],
        "gt_size_test": [],
        "l_size_test": [],
        "r_size_test": [],
    }

    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_size=1, shuffle=True, stable_left=True)
            
    augPath = os.path.join(input_dir, "left_aug.csv")
    if os.path.exists(augPath):
        print("Augmenting left column...")
        X_aug = pd.read_csv(augPath)
        X_aug = X_aug.head(int(augqty * len(X_aug)))
        
    X_test, y_test = X, y

    trainBegin = time()
    model.fit(X_train, id_column="id", left_aug=X_aug)
    trainEnd = time()
    results["train_scores"].append(model.evaluate(y_train, model.train_results_))
    results["train_times"].append(trainEnd-trainBegin)
            
    testBegin = time()
    y_pred = model.predict(X_test, id_column="id", left_aug=X_aug)
    testEnd = time()
    results["test_scores"].append(model.evaluate(y_test, y_pred))
    results["test_times"].append(testEnd-testBegin)
            
    results["gt_size_test"].append(len(y_test)) 
    results["gt_size_train"].append(len(y_train))
    results["l_size_test"].append(len(X_test[0]))
    results["l_size_train"].append(len(X_train[0]))
    results["r_size_test"].append(len(X_test[1]))
    results["r_size_train"].append(len(X_train[1]))
    results["train_precision_est"].append(model.precision_est_)

    filePath = os.path.join(result_dir, dataset)
    pathlib.Path(filePath).mkdir(parents=True, exist_ok=True)
    
    print(results)
    summary = pd.DataFrame(results, index=None)
    summary.drop(['train_scores', 'test_scores'], axis=1, inplace=True)
    train_df = pd.DataFrame(results["train_scores"])
    train_df.columns = [ f"{col}_train" for col in train_df.columns ]
    test_df = pd.DataFrame(results["test_scores"])
    test_df.columns = [ f"{col}_test" for col in test_df.columns ]
    summary = pd.concat([summary, train_df, test_df], axis=1)

    summary.to_csv(os.path.join(filePath, f"summary.csv"), index=False)
    model.save_model(os.path.join(filePath, f"{dataset}_{name}_model.pkl"))


@cli.command()
@click.argument("input-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("result-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dataset", type=click.STRING)
@click.argument("bm_pipeline", type=click.STRING)
@click.argument("attempt", type=click.STRING)
@click.option("--verbose", type=click.BOOL, default=False)
@click.option("--jfs", type=click.STRING, default="autofj_sm")
def run_benchmark(input_dir, result_dir, dataset, bm_pipeline, attempt, verbose, jfs):

    tmpDirs = glob("tmp*", recursive=False)
    for tmpDir in tmpDirs: 
        shutil.rmtree(tmpDir)
        
    data_l = pd.read_csv(os.path.join(input_dir, "left.csv"))
    data_r = pd.read_csv(os.path.join(input_dir, "right.csv"))
    data_gt = pd.read_csv(os.path.join(input_dir, "gt.csv"))

    outDir = os.path.join(result_dir, dataset, attempt)
    pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)

    X, y = (data_l, data_r), data_gt

    results = {
        "train_times": [],
        "test_times": [],
        "train_scores": [],
        "test_scores": [],
        "train_precision_est": [],
        "gt_size_train": [],
        "l_size_train": [],
        "r_size_train": [],
        "gt_size_test": [],
        "l_size_test": [],
        "r_size_test": [],
    }

    if bm_pipeline == "cv":
        id_column="id"
        cv=5 
        stable_left=True
        on = None

        kfold = KFold(n_splits=5, random_state=None, shuffle=True)
        
        results["fold"] = np.arange(1, cv+1, step=1).tolist()
            
        cv_itr = 0    
        for train, test in tqdm(kfold.split(X, y, stable_left=stable_left), total=cv, unit="fold"):
            X_train, y_train = train
            X_test, y_test = test
            cv_itr += 1

            augPath = os.path.join(input_dir, "left_aug.csv")
            X_aug = pd.read_csv(augPath) if os.path.exists(augPath) else None

            model = AutoFJ(precision_target=0.9, verbose=verbose, join_function_space=jfs)

            trainBegin = time()
            model.fit(X_train, y_train, id_column=id_column, on=on, left_aug=X_aug)
            trainEnd = time()
            model.save_model(os.path.join(outDir, f"{dataset}_{bm_pipeline}_{cv_itr}_model.pkl"))

            results["train_times"].append(trainEnd-trainBegin)
            results["train_scores"].append(model.evaluate(y_train, model.train_results_))

            testBegin = time()
            y_pred = model.predict(X_test, id_column=id_column, on=on, left_aug=X_aug)
            testEnd = time()

            results["test_times"].append(testEnd-testBegin)
            results["test_scores"].append(model.evaluate(y_test, y_pred))

            results["gt_size_test"].append(len(y_test)) 
            results["gt_size_train"].append(len(y_train))
            results["l_size_test"].append(len(X_test[0]))
            results["l_size_train"].append(len(X_train[0]))
            results["r_size_test"].append(len(X_test[1]))
            results["r_size_train"].append(len(X_train[1]))
            results["train_precision_est"].append(model.precision_est_)
    
    elif bm_pipeline == "complete-1":
        
        splitRates = [ 0.25, 0.50, 0.75, 1 ]
        results["split"] = splitRates

        for splitRate in splitRates:
            (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_size=splitRate, shuffle=True, stable_left=True)
            
            augPath = os.path.join(input_dir, "left_aug.csv")
            X_aug = pd.read_csv(augPath) if os.path.exists(augPath) else None

            X_test, y_test = X, y
            model = AutoFJ(precision_target=0.9, verbose=verbose, join_function_space=jfs)

            trainBegin = time()
            model.fit(X_train, id_column="id", left_aug=X_aug)
            trainEnd = time()
            results["train_scores"].append(model.evaluate(y_train, model.train_results_))
            results["train_times"].append(trainEnd-trainBegin)
            model.save_model(os.path.join(outDir, f"{dataset}_{bm_pipeline}_{splitRate}_model.pkl"))
            
            testBegin = time()
            y_pred = model.predict(X_test, id_column="id", left_aug=X_aug)
            testEnd = time()
            results["test_scores"].append(model.evaluate(y_test, y_pred))
            results["test_times"].append(testEnd-testBegin)

            results["gt_size_test"].append(len(y_test)) 
            results["gt_size_train"].append(len(y_train))
            results["l_size_test"].append(len(X_test[0]))
            results["l_size_train"].append(len(X_train[0]))
            results["r_size_test"].append(len(X_test[1]))
            results["r_size_train"].append(len(X_train[1]))
            results["train_precision_est"].append(model.precision_est_)
    
    elif bm_pipeline == "complete-2":

        augQtys = [0, 0.70, 1 ]
        results["l_aug_size"] = augQtys

        for augQty in augQtys:
            (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_size=1, shuffle=True, stable_left=True)
            
            augPath = os.path.join(input_dir, "left_aug.csv")
            X_aug = None
            if os.path.exists(augPath):
                X_aug = pd.read_csv(augPath)
                X_aug = X_aug.head(int(augQty * len(X_aug)))
                
            X_test, y_test = X, y
            model = AutoFJ(precision_target=0.9, verbose=verbose, join_function_space=jfs)

            trainBegin = time()
            model.fit(X_train, id_column="id", left_aug=X_aug)
            trainEnd = time()
            results["train_scores"].append(model.evaluate(y_train, model.train_results_))
            results["train_times"].append(trainEnd-trainBegin)
            model.save_model(os.path.join(outDir, f"{dataset}_{bm_pipeline}_{augQty}_model.pkl"))
            
            testBegin = time()
            y_pred = model.predict(X_test, id_column="id", left_aug=X_aug)
            testEnd = time()
            results["test_scores"].append(model.evaluate(y_test, y_pred))
            results["test_times"].append(testEnd-testBegin)
            
            results["gt_size_test"].append(len(y_test)) 
            results["gt_size_train"].append(len(y_train))
            results["l_size_test"].append(len(X_test[0]))
            results["l_size_train"].append(len(X_train[0]))
            results["r_size_test"].append(len(X_test[1]))
            results["r_size_train"].append(len(X_train[1]))
            results["train_precision_est"].append(model.precision_est_)
        
    elif bm_pipeline == "blocker":
        blockers = {
            "autofj_blocker": AutoFJBlocker(),
            "knowmore_blocker": KnowMoreBlocker(),
            "wikidata_blocker": WikidataBlocker(), 
        }

        results["blocker"] = list(blockers.values())

        for blocker_name, blocker_func in blockers.items():
            print(f"Using '{blocker_name}'...")
            model = AutoFJ(precision_target=0.9, verbose=False, join_function_space="autofj_sm", blocker=blocker_func)

            (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_size=1, shuffle=True, stable_left=True)
            
            augPath = os.path.join(input_dir, "left_aug.csv")
            X_aug = pd.read_csv(augPath) if os.path.exists(augPath) else None
                
            X_test, y_test = X, y
            model = AutoFJ(precision_target=0.9, verbose=verbose, join_function_space=jfs)

            trainBegin = time()
            model.fit(X_train, id_column="id", left_aug=X_aug)
            trainEnd = time()
            results["train_scores"].append(model.evaluate(y_train, model.train_results_))
            results["train_times"].append(trainEnd-trainBegin)
            model.save_model(os.path.join(outDir, f"{dataset}_{bm_pipeline}_{blocker_name}_model.pkl"))

            testBegin = time()
            y_pred = model.predict(X_test, id_column="id", left_aug=X_aug)
            testEnd = time()
            results["test_scores"].append(model.evaluate(y_test, y_pred))
            results["test_times"].append(testEnd-testBegin)

            results["gt_size_test"].append(len(y_test)) 
            results["gt_size_train"].append(len(y_train))
            results["l_size_test"].append(len(X_test[0]))
            results["l_size_train"].append(len(X_train[0]))
            results["r_size_test"].append(len(X_test[1]))
            results["r_size_train"].append(len(X_train[1]))
            results["train_precision_est"].append(model.precision_est_)

    else: 
        raise RuntimeError(f"Unknown pipeline '{bm_pipeline}'")
    
    print(results)
    summary = pd.DataFrame(results, index=None)
    summary.drop(['train_scores', 'test_scores'], axis=1, inplace=True)
    train_df = pd.DataFrame(results["train_scores"])
    train_df.columns = [ f"{col}_train" for col in train_df.columns ]
    test_df = pd.DataFrame(results["test_scores"])
    test_df.columns = [ f"{col}_test" for col in test_df.columns ]
    summary = pd.concat([summary, train_df, test_df], axis=1)

    summary.to_csv(os.path.join(outDir, f"summary_{bm_pipeline}.csv"), index=False)

if __name__ == '__main__':
    cli()