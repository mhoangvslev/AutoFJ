import os
import pandas as pd

rule all:
    input: 
        # For each benchmark folder, we will run AutoFJ on 3 configurations:
        #     1. 90% train rate
        #     2. 75% train rate
        #     3. 50% train rate
        #     4. 25% train rate
        #     5. 5-fold cross-validation
        expand(
            "{resultDir}/summary/{bm_pipeline}/{phase}_result.csv", 
            resultDir=config["resultDir"],
            bm_pipeline=["90", "75", "50", "25", "cv"],
            phase=["train", "test"]
        )

rule autofj_benchmark_summary:
    input: 
        expand(
            "{{resultDir}}/{dataset}/{dataset}_{{bm_pipeline}}_model.pkl",
            dataset=sorted(os.listdir(config["dataDir"]))
        )
    output: "{resultDir}/summary/{bm_pipeline}/{phase}_result.csv"
    run:
        df = pd.read_csv(f"{wildcards.resultDir}/{wildcards.dataset}/{wildcards.bm_pipeline}/{wildcards.phase}_result.csv")
        df["dataset"] = wildards.dataset
        fileName = f"{wildards.resultDir}/{wildcards.phase}_result.csv"
        isNew = os.path.exists(fileName)
        df.to_csv(fileName, index=False, header=isNew, mode="w" if isNew else "a")

rule autofj_benchmark:
    # For each benchmark dataset:
    #     1. Load left, right and ground truth tables
    #     2. Feed them to the script that run the benchmark
    input: 
        left = "src/autofj/benchmark/{dataset}/left.csv",
        right = "src/autofj/benchmark/{dataset}/right.csv",
        gt = "src/autofj/benchmark/{dataset}/gt.csv",
    output: "{resultDir}/{dataset}/{dataset}_{bm_pipeline}_model.pkl"
    shell:
        "python scripts/autofj_benchmark.py autofj-benchmark-cv {input.left} {input.right} {input.gt} {wildcards.resultDir} {wildcards.dataset} {wildcards.bm_pipeline}"