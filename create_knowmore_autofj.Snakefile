rule all:
    input: 
        expand(
            "{dataDir}/KM_{dataset}_{variant}/left_aug.csv", 
            dataDir=config["dataDir"],
            dataset=["Book", "Movie"],
            variant=["full", "1col", "2col"],
        )

rule make_subset:
    input: "{dataDir}/{dataset}_wiki_all.csv", "{dataDir}/KM_{dataset}_{variant}/{dataset}_em_gt.json"
    output: "{dataDir}/KM_{dataset}_{variant}/left.csv"
    shell: "python scripts/make_dataset.py make-dataset {wildcards.dataDir} {wildcards.dataset} {wildcards.variant}"

rule augment_left:
    input:  "{dataDir}/KM_{dataset}_{variant}/left.csv"
    output: "{dataDir}/KM_{dataset}_{variant}/left_aug.csv"
    shell: "python scripts/make_dataset.py augment-left {wildcards.dataDir} {wildcards.dataset} {wildcards.variant}"

rule get_all_wikidata:
    output: "{dataDir}/{dataset}_wiki_all.csv"
    shell: "python scripts/make_dataset.py get-all-wikidata {wildcards.dataDir} {wildcards.dataset}"

rule download_ressources:
    output: "{dataDir}/KM_{dataset}_{variant}/{dataset}_em_gt.json"
    shell: "wget http://l3s.de/~yu/knowmore/groundtruth/knowmore_match_gt/{wildcards.dataset}_em_gt.json -O {wildcards.dataDir}/KM_{wildcards.dataset}_{wildcards.variant}/{wildcards.dataset}_em_gt.json"