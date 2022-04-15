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
            "{{resultDir}}/{dataset}/{dataset}_{bm_pipeline}_model.pkl",
            dataset=sorted(os.listdir(config["dataDir"])),
            bm_pipeline=["100", "90", "75", "50", "25", "cv"],
        )
    output: "{resultDir}/cross_validation_scores.csv"
    run:
        for model in input:
            homeDir = os.path.dirname(model)
            dataset = os.path.basename(homeDir)
            modelName = os.path.basename(model)

            bm_pipeline = re.sub(r'\w+_(\w+)_model\.pkl', r'\1', modelName)

            for phase in ["train", "test"]:
                df = pd.read_csv(os.path.join(homeDir, f"{bm_pipeline}", f"{phase}_scores.csv"))
                df.drop("Unnamed: 0", axis=1, inplace=True)
                df["dataset"] = dataset
                df["phase"] = phase

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
    output: "{resultDir}/{dataset}/{dataset}_{bm_pipeline}_model.pkl"
    shell:
        "python scripts/autofj_benchmark.py autofj-benchmark-cv {input.left} {input.right} {input.gt} {wildcards.resultDir} {wildcards.dataset} {wildcards.bm_pipeline}"