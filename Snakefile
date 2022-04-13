import os

rule all:
    input: 
        # For each benchmark folder, we will run AutoFJ on 3 configurations:
        #     1. 90% train rate
        #     2. 75% train rate
        #     3. 50% train rate
        #     4. 25% train rate
        #     5. 5-fold cross-validation
        expand(
            "{resultDir}/{benchmark}/{benchmark}_{bm_pipeline}_model.pkl", 
            resultDir=config["resultDir"],
            benchmark=os.listdir(config["dataDir"]),
            bm_pipeline=["90", "75", "50", "25", "cv"]
        )

rule autofj_benchmark:
    # For each benchmark dataset:
    #     1. Load left, right and ground truth tables
    #     2. Feed them to the script that run the benchmark
    input: 
        left = "src/autofj/benchmark/{benchmark}/left.csv",
        right = "src/autofj/benchmark/{benchmark}/right.csv",
        gt = "src/autofj/benchmark/{benchmark}/gt.csv",
    output: "{resultDir}/{benchmark}/{benchmark}_{bm_pipeline}_model.pkl"
    shell:
        "python scripts/autofj_benchmark.py autofj-benchmark-cv {input.left} {input.right} {input.gt} {wildcards.resultDir} {wildcards.benchmark} {wildcards.bm_pipeline}"

