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
        train, test = train_test_split(X, y, train_size=splitRate, shuffle=False, stable_left=True)

        X_train, y_train = train
        X_test, y_test = test

        results = {}

        model.fit(X_train, y_train, id_column="id", on=["title"])
        results['train_scores'] = model.evaluate(y_train, model.train_results_)
        y_pred = model.predict(X_test, id_column="id", on=["title"])
        results["test_scores"] = model.evaluate(y_test, y_pred)

    for category, result in results.items():
        try:
            df = pd.DataFrame(result)
        except ValueError:
            df = pd.DataFrame(result, index=[0])
        
        filePath = os.path.join(result_dir, dataset, bm_pipeline)
        pathlib.Path(filePath).mkdir(parents=True, exist_ok=True)
        df.to_csv(os.path.join(filePath, f"{category}.csv"))

    model.save_model(os.path.join(result_dir, dataset, f"{dataset}_{bm_pipeline}_model.pkl"))

if __name__ == '__main__':
    cli()