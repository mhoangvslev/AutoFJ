import os
import pandas as pd
import re

rule all:
    input: 
        # For each benchmark folder, we will run AutoFJ on 3 configurations:
        #     1. 90% train rate
        #     2. 75% train rate
        #     3. 50% train rate
        #     4. 25% train rate
        #     5. 5-fold cross-validation
        expand(
            "{resultDir}/cross_validation_scores.csv", 
            resultDir=config["resultDir"],
            phase=["train", "test"]
        )

rule autofj_benchmark_summary:
    input: 
        expand(
            "{{resultDir}}/{dataset}/{attempt}/summary_{bm_pipeline}.csv",
            dataset=sorted(os.listdir(config["dataDir"])),
            attempt=range(3),
            bm_pipeline=["100", "90", "75", "50", "25", "cv"],
        )
    output: "{resultDir}/cross_validation_scores.csv"
    run:
        for summaryFile in input:
            pattern = r"/(\w+)/(\d+)/summary_(\w+).csv"
            match = re.search(pattern, summaryFile)
            dataset = match.group(1)
            attempt = match.group(2)
            bm_pipeline = match.group(3)

            summary_fn = os.path.basename(summaryFile)

            df = pd.read_csv(summaryFile)
            df["dataset"] = dataset
            df["attempt"] = attempt

            if bm_pipeline == "cv":
                fileName = f"{wildcards.resultDir}/cross_validation_scores.csv"
            else:
                df["trainSize"] = bm_pipeline
                fileName = f"{wildcards.resultDir}/scalability_scores.csv"
            isNew = not os.path.exists(fileName)
            df.to_csv(fileName, header=isNew, index=False, mode="w" if isNew else "a")                

def get_dataset(wildcards):
    return {
        "left": os.path.join(config["dataDir"], wildcards.dataset, "left.csv"),
        "right": os.path.join(config["dataDir"], wildcards.dataset, "right.csv"),
        "gt": os.path.join(config["dataDir"], wildcards.dataset, "gt.csv")
    }

rule autofj_benchmark:
    # For each benchmark dataset:
    #     1. Load left, right and ground truth tables
    #     2. Feed them to the script that run the benchmark
    input: 
        unpack(get_dataset)
    output: "{resultDir}/{dataset}/{attempt}/summary_{bm_pipeline}.csv"
    shell:
        "python scripts/autofj_benchmark.py autofj-benchmark-cv {input.left} {input.right} {input.gt} {wildcards.resultDir} {wildcards.dataset} {wildcards.bm_pipeline} {wildcards.attempt}"