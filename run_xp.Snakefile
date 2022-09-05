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
            "{resultDir}/benchmark_{bm_pipeline}.csv", 
            resultDir=config["resultDir"],
            phase=["train", "test"],
            bm_pipeline=["complete-2"] # Choose from ["complete-1", "complete-2", "cv", "blocker"]
        )

rule autofj_benchmark_prediction:
    input: 
        expand(
            "{{resultDir}}/{dataset}/{attempt}/{dataset}_{{bm_pipeline}}_model.pkl",
            dataset=sorted(os.listdir(config["dataDir"])),
            attempt=range(3)
        )
    output:
        "{resultDir}/{dataset}/{attempt}/pred.csv"
    run:
        "python scripts/autofj_benchmark.py predict {input} {input.gt} {wildcards.resultDir} {wildcards.dataset} {wildcards.bm_pipeline} {wildcards.attempt}"

rule autofj_benchmark_summary:
    input: 
        expand(
            "{{resultDir}}/{dataset}/{attempt}/summary_{{bm_pipeline}}.csv",
            dataset=sorted(os.listdir(config["dataDir"])),
            attempt=range(3)
        )
    output: "{resultDir}/benchmark_{bm_pipeline}.csv"
    run:
        for summaryFile in input:
            pattern = r"/(\w+)/(\d+)/summary_(.*).csv"
            match = re.search(pattern, summaryFile)
            dataset = match.group(1)
            attempt = match.group(2)
            bm_pipeline = match.group(3)

            summary_fn = os.path.basename(summaryFile)

            df = pd.read_csv(summaryFile)
            df["dataset"] = dataset
            df["attempt"] = attempt

            fileName = f"{wildcards.resultDir}/benchmark_{wildcards.bm_pipeline}.csv"
            isNew = not os.path.exists(fileName)
            df.to_csv(fileName, header=isNew, index=False, mode="w" if isNew else "a") 

            # Predict
            homeDir = os.path.join(wildcards.resultDir, dataset, attempt)
            modelName = f"{dataset}_{bm_pipeline}_model.pkl"
            modelPath = os.path.join(homeDir, modelName)

            dataPath = os.path.join(config["dataDir"], dataset)
            subprocess.run(["python", "scripts/autofj_benchmark.py", "predict", modelPath, dataPath, f"--outdir={homeDir}"])          

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
        "python scripts/autofj_benchmark.py autofj-benchmark {input.left} {input.right} {input.gt} {wildcards.resultDir} {wildcards.dataset} {wildcards.bm_pipeline} {wildcards.attempt}"