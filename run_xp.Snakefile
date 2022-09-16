import os
import pandas as pd
import re
import subprocess

rule all:
    input: 
        expand(
            "{resultDir}/benchmark_{bm_pipeline}.csv",
            resultDir=config["resultDir"],
            bm_pipeline=["complete-2"] # Choose from ["complete-1", "complete-2", "cv", "blocker"]
        )

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

            fileName = f"{wildcards.resultDir}/benchmark_{bm_pipeline}.csv"
            isNew = not os.path.exists(fileName)
            df.to_csv(fileName, header=isNew, index=False, mode="w" if isNew else "a") 
        
        # Predict
        homeDir = os.path.join(wildcards.resultDir, dataset, attempt)

        for f in os.listdir(homeDir):
            if f.endswith("_model.pkl"):
                dataPath = os.path.join(config["dataDir"], dataset)
                outDir = os.path.join(homeDir, f.replace(".pkl", ""))
                os.makedirs(outDir, exist_ok=True)
                modelPath = os.path.join(homeDir, f)
                subprocess.run(["python", "scripts/autofj_benchmark.py", "predict", modelPath, dataPath, f"--outdir={outDir}"])   

rule autofj_benchmark:
    # For each benchmark dataset:
    #     1. Load left, right and ground truth tables
    #     2. Feed them to the script that run the benchmark
    output: "{resultDir}/{dataset}/{attempt}/summary_{bm_pipeline}.csv"
    shell:
        "python scripts/autofj_benchmark.py run-benchmark {config[dataDir]}/{wildcards.dataset} {wildcards.resultDir} {wildcards.dataset} {wildcards.bm_pipeline} {wildcards.attempt}"
