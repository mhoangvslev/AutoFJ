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
@click.option("--outDir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
def predict(modelname, dataset, id, on, outDir):
    model: AutoFJ = AutoFJ.load_model(modelname)
    data_l = pd.read_csv(os.path.join(dataset, "left.csv"))
    data_r = pd.read_csv(os.path.join(dataset, "right.csv"))
    data_gt = pd.read_csv(os.path.join(dataset, "gt.csv"))

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

    print(f"Test results: {test_results}")
    print(f"True positive: {tp}")
    print(f"False positive: {fp}")
    print(f"False negative: {fn}")

    if outDir is None: outDir = dataset

    y_pred.to_csv(os.path.join(outDir, "pred.csv"), index=False)
    tp.to_csv(os.path.join(outDir, "pred_tp.csv"), index=False)
    fp.to_csv(os.path.join(outDir, "pred_fp.csv"), index=False)
    fn.to_csv(os.path.join(outDir, "pred_fn.csv"), index=False)


@cli.command()
@click.argument("left", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("right", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("gt", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("result-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dataset", type=click.STRING)
@click.argument("bm_pipeline", type=click.STRING)
@click.argument("attempt", type=click.STRING)
def autofj_benchmark(left, right, gt, result_dir, dataset, bm_pipeline, attempt):

    tmpDirs = glob("tmp*", recursive=False)
    for tmpDir in tmpDirs: 
        shutil.rmtree(tmpDir)
        
    data_l = pd.read_csv(left)
    data_r = pd.read_csv(right)
    data_gt = pd.read_csv(gt)

    X, y = (data_l, data_r), data_gt

    model = AutoFJ(precision_target=0.9, verbose=False, join_function_space="autofj_sm")

    results = {
        "train_times": [],
        "test_times": [],
        "train_scores": [],
        "test_scores": [],
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
            
        for train, test in tqdm(kfold.split(X, y, stable_left=stable_left), total=cv, unit="fold"):
            X_train, y_train = train
            X_test, y_test = test

            homeDir = os.path.dirname(os.path.realpath(left))
            augPath = os.path.join(homeDir, "left_aug.csv")
            if os.path.exists(augPath):
                print("Augmenting left column...")
                X_aug = pd.read_csv(augPath)
                X_train_l_org, X_train_r = X_train
                X_train_l = X_train_l_org.append(X_aug[X_train_l_org.columns])
                X_train = (X_train_l, X_train_r)
                print(f"X_train (original): {len(X_train_l_org)}; left_aug: {len(X_aug)}; X_train (augmented): {len(X_train_l)}")

            trainBegin = time()
            model.fit(X_train, y_train, id_column=id_column, on=on)
            trainEnd = time()

            results["train_times"].append(trainEnd-trainBegin)
            results["train_scores"].append(model.evaluate(y_train, model.train_results_))

            testBegin = time()
            y_pred = model.predict(X_test, id_column=id_column, on=on)
            testEnd = time()

            results["test_times"].append(testEnd-testBegin)
            results["test_scores"].append(model.evaluate(y_test, y_pred))

            results["gt_size_test"].append(len(y_test)) 
            results["gt_size_train"].append(len(y_train))
            results["l_size_test"].append(len(X_test[0]))
            results["l_size_train"].append(len(X_train[0]))
            results["r_size_test"].append(len(X_test[1]))
            results["r_size_train"].append(len(X_train[1]))
    
    elif bm_pipeline == "complete-1":
        
        splitRates = [ 0.25, 0.50, 0.75, 1 ]
        results["split"] = splitRates

        for splitRate in splitRates:
            (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_size=splitRate, shuffle=True, stable_left=True)
            
            homeDir = os.path.dirname(os.path.realpath(left))
            augPath = os.path.join(homeDir, "left_aug.csv")
            if os.path.exists(augPath):
                print("Augmenting left column...")
                X_aug = pd.read_csv(augPath)
                X_train_l_org, X_train_r = X_train
                X_train_l = X_train_l_org.append(X_aug[X_train_l_org.columns])
                X_train = (X_train_l, X_train_r)
                print(f"X_train (original): {len(X_train_l_org)}; left_aug: {len(X_aug)}; X_train (augmented): {len(X_train_l)}")
            
            X_test, y_test = X, y

            trainBegin = time()
            model.fit(X_train, id_column="id")
            trainEnd = time()
            results["train_scores"].append(model.evaluate(y_train, model.train_results_))
            results["train_times"].append(trainEnd-trainBegin)
            
            testBegin = time()
            y_pred = model.predict(X_test, id_column="id")
            testEnd = time()
            results["test_scores"].append(model.evaluate(y_test, y_pred))
            results["test_times"].append(testEnd-testBegin)

            results["gt_size_test"].append(len(y_test)) 
            results["gt_size_train"].append(len(y_train))
            results["l_size_test"].append(len(X_test[0]))
            results["l_size_train"].append(len(X_train[0]))
            results["r_size_test"].append(len(X_test[1]))
            results["r_size_train"].append(len(X_train[1]))
    
    elif bm_pipeline == "complete-2":

        augQtys = [0, 50, 100, 1000, 2000, 3000, 4000, 5000 ]
        results["l_aug_size"] = augQtys

        for augQty in augQtys:
            (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_size=1, shuffle=True, stable_left=True)
            
            homeDir = os.path.dirname(os.path.realpath(left))
            augPath = os.path.join(homeDir, "left_aug.csv")
            if os.path.exists(augPath):
                print("Augmenting left column...")
                X_aug = pd.read_csv(augPath).head(augQty)
                X_train_l_org, X_train_r = X_train
                X_train_l = X_train_l_org.append(X_aug[X_train_l_org.columns])
                X_train = (X_train_l, X_train_r)
                print(f"X_train (original): {len(X_train_l_org)}; left_aug: {len(X_aug)}; X_train (augmented): {len(X_train_l)}")
            
            X_test, y_test = X, y

            trainBegin = time()
            model.fit(X_train, id_column="id")
            trainEnd = time()
            results["train_scores"].append(model.evaluate(y_train, model.train_results_))
            results["train_times"].append(trainEnd-trainBegin)
            
            testBegin = time()
            y_pred = model.predict(X_test, id_column="id")
            testEnd = time()
            results["test_scores"].append(model.evaluate(y_test, y_pred))
            results["test_times"].append(testEnd-testBegin)
            
            results["gt_size_test"].append(len(y_test)) 
            results["gt_size_train"].append(len(y_train))
            results["l_size_test"].append(len(X_test[0]))
            results["l_size_train"].append(len(X_train[0]))
            results["r_size_test"].append(len(X_test[1]))
            results["r_size_train"].append(len(X_train[1]))
        
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
            
            homeDir = os.path.dirname(os.path.realpath(left))
            augPath = os.path.join(homeDir, "left_aug.csv")
            if os.path.exists(augPath):
                print("Augmenting left column...")
                X_aug = pd.read_csv(augPath).head(augQty)
                X_train_l_org, X_train_r = X_train
                X_train_l = X_train_l_org.append(X_aug[X_train_l_org.columns])
                X_train = (X_train_l, X_train_r)
                print(f"X_train (original): {len(X_train_l_org)}; left_aug: {len(X_aug)}; X_train (augmented): {len(X_train_l)}")
            
            X_test, y_test = X, y

            trainBegin = time()
            model.fit(X_train, id_column="id")
            trainEnd = time()
            results["train_scores"].append(model.evaluate(y_train, model.train_results_))
            results["train_times"].append(trainEnd-trainBegin)
            
            testBegin = time()
            y_pred = model.predict(X_test, id_column="id")
            testEnd = time()
            results["test_scores"].append(model.evaluate(y_test, y_pred))
            results["test_times"].append(testEnd-testBegin)

            results["gt_size_test"].append(len(y_test)) 
            results["gt_size_train"].append(len(y_train))
            results["l_size_test"].append(len(X_test[0]))
            results["l_size_train"].append(len(X_train[0]))
            results["r_size_test"].append(len(X_test[1]))
            results["r_size_train"].append(len(X_train[1]))

    else: 
        raise RuntimeError(f"Unknown pipeline '{bm_pipeline}'")

    filePath = os.path.join(result_dir, dataset, attempt)
    pathlib.Path(filePath).mkdir(parents=True, exist_ok=True)
    
    print(results)
    summary = pd.DataFrame(results, index=None)
    summary.drop(['train_scores', 'test_scores'], axis=1, inplace=True)
    train_df = pd.DataFrame(results["train_scores"])
    train_df.columns = [ f"{col}_train" for col in train_df.columns ]
    test_df = pd.DataFrame(results["test_scores"])
    test_df.columns = [ f"{col}_test" for col in test_df.columns ]
    summary = pd.concat([summary, train_df, test_df], axis=1)

    summary.to_csv(os.path.join(filePath, f"summary_{bm_pipeline}.csv"), index=False)
    model.save_model(os.path.join(filePath, f"{dataset}_{bm_pipeline}_model.pkl"))

if __name__ == '__main__':
    cli()