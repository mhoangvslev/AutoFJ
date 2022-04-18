from glob import glob
import pathlib
import shutil
from autofj import AutoFJ
import pandas as pd
import os
import click

from autofj import cross_validate, train_test_split
from sklearn.utils import shuffle

@click.group()
def cli():
    pass

@cli.command()
@click.argument("left", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("right", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("gt", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("result-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dataset", type=click.STRING)
@click.argument("bm_pipeline", type=click.STRING)
def autofj_benchmark_cv(left, right, gt, result_dir, dataset, bm_pipeline):

    tmpDirs = glob("tmp*", recursive=False)
    for tmpDir in tmpDirs: 
        shutil.rmtree(tmpDir)
        
    left = pd.read_csv(left)
    right = pd.read_csv(right)
    gt = pd.read_csv(gt)

    X = (left, right)
    y = gt

    model = AutoFJ(precision_target=0.9, verbose=False)
    if bm_pipeline == "cv":
        results = cross_validate(model, X, y, id_column="id", on=["title"], cv=5, scorer=model.evaluate, stable_left=True)
    else:
        splitRate = float(bm_pipeline) * 0.01
        (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_size=splitRate, shuffle=False, stable_left=True)

        results = {}

        model.fit(X_train, id_column="id", on=["title"])
        results["train_scores"] = model.evaluate(y_train, model.train_results_) 
        y_pred = model.predict(X, id_column="id", on=["title"])
        results["test_scores"] = model.evaluate(y, y_pred)

        results["gt_size_test"], results["gt_size_train"] = len(y_test), len(y_train) 
        results["l_size_test"], results["l_size_train"] = len(X_test[0]), len(X_train[0]) 
        results["r_size_test"], results["r_size_train"] = len(X_test[1]), len(X_train[1])

    df = pd.DataFrame(results, index=None if bm_pipeline == "cv" else [0])

    df.drop(['train_scores', 'test_scores'], axis=1, inplace=True)
    train_df = pd.DataFrame(results["train_scores"], index=None if bm_pipeline == "cv" else [0])
    train_df.columns = [ f"{col}_train" for col in train_df.columns ]
    test_df = pd.DataFrame(results["test_scores"], index=None if bm_pipeline == "cv" else [0])
    test_df.columns = [ f"{col}_test" for col in test_df.columns ]
    df = pd.concat([df, train_df, test_df], axis=1)

    filePath = os.path.join(result_dir, dataset)
    pathlib.Path(filePath).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(filePath, f"summary_{bm_pipeline}.csv"), index=False)

    model.save_model(os.path.join(result_dir, dataset, f"{dataset}_{bm_pipeline}_model.pkl"))

if __name__ == '__main__':
    cli()