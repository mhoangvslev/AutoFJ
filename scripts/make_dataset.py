import json
from langdetect import detect
import click
import pandas as pd
from dateutil import parser
from scipy import spatial
import numpy as np
import os
import requests
from qwikidata.sparql import return_sparql_query_results
from tqdm import tqdm
import re

from string import punctuation
import spacy

import spacy_universal_sentence_encoder
nlp = spacy_universal_sentence_encoder.load_model('xx_use_lg')


@click.group()
def cli():
    pass


WD_INST = {
    "Book": "Q7725634",
    "Movie": "Q11424"
}

MEDIAWIKI_CACHE = {}
PROP_CACHE = {}


def normalize_ent(wikidata_aux):
    entityDesc = {}
    for propId, propDesc in wikidata_aux["claims"].items():
        propName = get_property_label(propId)

        propVals = [
            normalize_prop(p["mainsnak"]['datavalue']['value'])
            for p in propDesc
        ]

        entityDesc[propName] = propVals[0] if len(
            propVals) == 1 else "|".join(map(str, propVals))
    try:
        entityDesc['title'] = wikidata_aux["labels"]["en"]["value"]
    except:
        print(wikidata_aux)

    return entityDesc


def normalize_prop(propVal):
    if not isinstance(propVal, str):
        if propVal.get("entity-type") == "item":
            entityId = propVal.get("id")
            return get_property_label(entityId)
        elif propVal.get("time") is not None:
            time = propVal.get("time")
            # return parser.parse(time)
            return re.search(r"^\+?(\d{4})", time).group(1)
        elif propVal.get("latitude") is not None:
            return f"({propVal.get('latitude')}, {propVal.get('longitude')})"
        elif propVal.get("text") is not None:
            return propVal.get("text")
        elif propVal.get("unit") is not None:
            unit = get_property_label(
                re.search(r"/(\w+)$", propVal.get("unit")).group(1))
            amount = propVal.get("amount")
            return f"{amount} {unit}"
        elif propVal.get("mainsnak") is not None:
            return propVal["mainsnak"]['datavalue']['value']
    return propVal


def get_wikidata_ent_by_title(title):
    if MEDIAWIKI_CACHE.get(title) is None:
        res = requests.get(
            url="https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "sites": "enwiki",
                "format": "json",
                "titles": title,
                "normalize": True
            }).json()
        ent_id = list(res["entities"].keys())[0]
        aux = res["entities"][ent_id]

        MEDIAWIKI_CACHE[title] = ent_id, aux
    return MEDIAWIKI_CACHE[title]


def get_wikidata_ent(id):
    if MEDIAWIKI_CACHE.get(id) is None:
        res = requests.get(
            url="https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "sites": "enwiki",
                "format": "json",
                "ids": id,
                "normalize": True
            }).json()
        ent_id = list(res["entities"].keys())[0]
        aux = res["entities"][ent_id]

        MEDIAWIKI_CACHE[id] = ent_id, aux
    return MEDIAWIKI_CACHE[id]


def get_property_label(propId):
    if PROP_CACHE.get(propId) is None:
        res = requests.get(
            url="https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": propId,
                "languages": "en",
                "props": "labels",
                "format": "json"
            }
        )

        labels = res.json()["entities"][propId]["labels"]
        if len(labels) == 0:
            _, ent = get_wikidata_ent(propId)

            if "en" in ent["claims"].keys():
                description = ent["claims"]["en"]["value"]
                alias = list(ent["aliases"].values())[0][0]["value"]
                PROP_CACHE[propId] = f"{alias} ({description})"
            else:
                PROP_CACHE[propId] = list(ent["labels"].values())[0]["value"]
        else:
            PROP_CACHE[propId] = res.json(
            )["entities"][propId]["labels"]["en"]["value"]
    return PROP_CACHE[propId]


def parse_results(result, qwikidata=True):
    entDesc = {}
    skipDict = {}

    entIdStr = 'entId' if qwikidata else '?entId'
    propNameStr = 'propName' if qwikidata else '?propName'
    propValueStr = 'propStr' if qwikidata else '?propValue'

    for b in result["results"]["bindings"]:

        entId = b[entIdStr]['value']
        # if entId in df["id"]: continue

        propName = b[propNameStr]['value']
        propValue = b[propValueStr]['value']

        entId = re.search("/(Q\d+)$", entId).group(1)

        if entDesc.get(entId) is None:
            entDesc[entId] = {}
            skipDict[entId] = {}

        # Handle skips and exceptions
        if propName in skipDict[entId]:
            continue
        if "http://www.wikidata.org/entity/statement/" in propValue:
            continue

        # Property name
        if re.match(r"^.*#(\w+)$", propName):
            propName = re.search(r"#(\w+)$", propName).group(1)
        elif re.match(r"^.*/(\w+)$", propName):
            propName = re.search(r"/(\w+)$", propName).group(1)
            if re.match(r"P\d+", propName):
                propName = get_property_label(propName)
        else:
            raise ValueError(propName)

        #if propName == "label": propName = "title"

        # Prop value
        if b[propValueStr]['type'] == "uri":
            try:
                entSearch = re.search("/(Q\d+)$", propValue)
                if entSearch:
                    propValue = get_property_label(entSearch.group(1))
            except:
                continue
        elif b[propValueStr]['type'] == "literal":
            # If there is a label in English
            try:
                lang = detect(propValue)
                if lang == "en":
                    skipDict[entId][propName] = propValue
            except:
                pass

        entDesc[entId][propName] = propValue
    return entDesc


def get_wikidata_ent_sparql(instance, keywords=[], qty=None):
    sparql_query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?entId ?propName ?propValue  WHERE {{
        hint:Query hint:optimizer "None" .  
        {{
            SELECT DISTINCT ?entId WHERE {{  
                #VALUES ?term {{{' '.join(map(repr, keywords))}}}
                ?entId  wdt:P31 wd:{WD_INST[instance]} .
                ?entId  (wdt:P1476 | rdfs:label) ?title .
                FILTER (LANG(?title) = "en")
                #FILTER (CONTAINS(LCASE(?title), ?term) && LANG(?title) = "en" )
            }}
        }} .
        ?entId  (wdt:P1476 | rdfs:label) ?title;
                ?propName ?propValue .
    }}
    """

    if qty is not None:
        sparql_query += f" LIMIT {qty}"

    res = return_sparql_query_results(sparql_query)
    return parse_results(res)


def word_embedding(word):
    return nlp(word).vector


@cli.command()
@click.argument("datadir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dataset", type=click.STRING)
def get_all_wikidata(datadir, dataset):
    json_path = os.path.join(datadir, f"{dataset}_wiki_all.json")
    if not os.path.exists(json_path):
        gdrive_id = None
        if dataset == "Book":
            gdrive_id = "1GEOKAR9BAQQnHGBdx-QH6AaK3CvjdLSh" 
        elif dataset == "Movie":
            gdrive_id = "1R91bhoJ41e9nwsD2D4by_Zop64ehTrB0" 
        
        os.system(f"gdown --id {gdrive_id} -O {json_path}")
    src_json = json.load(open(json_path, mode="r"))

    left_aug = pd.DataFrame()

    for entId, entAux in tqdm(parse_results(src_json, qwikidata=False).items()):
        # Augmentation
        title = get_property_label(entId)
        entAux["title (en)"] = title
        entry = pd.DataFrame(entAux, index=[entId])
        #entry["block"] = block
        left_aug = left_aug.append(entry)
    
    (
        left_aug
            .drop_duplicates()
            .dropna(how="all", subset=[c for c in left_aug.columns if "id" not in c])
            .to_csv(os.path.join(datadir, f"{dataset}_wiki_all.csv"), index=False)
    )

def align_attributes(datadir, dataset, variant):
    subset_mapping = {
        "left": os.path.join(datadir, f"KM_{dataset}_{variant}", "left.csv"),
        "right": os.path.join(datadir, f"KM_{dataset}_{variant}", "right.csv"),
        "gt": os.path.join(datadir, f"KM_{dataset}_{variant}", "gt.csv"),
        "left_aug": os.path.join(datadir, f"KM_{dataset}_{variant}", "left_aug.csv")
    }

    left = pd.read_csv(subset_mapping["left"])
    right = pd.read_csv(subset_mapping["left"])

    left_col_emb = {c: word_embedding(c) for c in left.columns}
    right_col_emb = {c: word_embedding(c) for c in right.columns}

    attrA = {}

    for col_l, emb_l in left_col_emb.items():
        for col_r, emb_r in right_col_emb.items():
            if attrA.get(col_l) is None:
                attrA[col_l] = []
            attrA[col_l].append({
                "candidate": col_r,
                "cosine": spatial.distance.cosine(emb_l, emb_r),
                "euclidean": spatial.distance.euclidean(emb_l, emb_r)
            })

        attrA[col_l] = pd.DataFrame(attrA[col_l]).set_index("candidate")

    alignments = pd.DataFrame(
        {k: [v["cosine"].idxmin(), v["cosine"].min()] for k, v in attrA.items()}).transpose()
    alignments.columns = ["candidate", "cosine"]
    alignments.sort_values(by="cosine", ascending=True, inplace=True)

    mapping = (
        alignments[alignments["cosine"] < 0.32]
               .drop_duplicates(subset="candidate", keep="first")
               .drop("cosine", axis=1)
               .squeeze()
               .to_dict()
    )

    return mapping

@cli.command()
@click.argument("datadir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dataset", type=click.STRING)
@click.argument("variant", type=click.STRING)
def make_dataset(datadir, dataset, variant):
    jsonFile = os.path.join(
        datadir, f"KM_{dataset}_{variant}", f"{dataset}_em_gt.json")
    knowmore = pd.read_json(jsonFile, lines=True)
    left = pd.DataFrame()
    right = pd.DataFrame()
    gt = pd.DataFrame()

    subset_mapping = {
        "left": os.path.join(datadir, f"KM_{dataset}_{variant}", "left.csv"),
        "right": os.path.join(datadir, f"KM_{dataset}_{variant}", "right.csv"),
        "gt": os.path.join(datadir, f"KM_{dataset}_{variant}", "gt.csv"),
        "left_aug": os.path.join(datadir, f"KM_{dataset}_{variant}", "left_aug.csv")
    }

    for _, row in tqdm(knowmore.iterrows()):
        desc = row['data']
        wikipage = desc["wikipage"]
        title = re.search(r"wiki/(.*)$", wikipage).group(1)
        if title == "The_Storyteller_(novel)":
            title = "Storyteller_(novel)"

        wikidata_ent, wikidata_aux = get_wikidata_ent_by_title(title)
        if wikidata_ent == "-1":
            print(wikidata_ent, wikipage, title)

        left = left.append(
            pd.DataFrame(normalize_ent(wikidata_aux), index=[wikidata_ent])
        )

        if isinstance(desc['objects'], str):
            desc['objects'] = desc['objects'].split('`')
            desc['predicates'] = desc['predicates'].split('`')

        right_entry = {}

        for i in range(len(desc["objects"])):
            pred = desc["predicates"][i]
            obj = desc["objects"][i]
            if right_entry.get(pred) is None:
                right_entry[pred] = []
            right_entry[pred].append(obj)

        right_entry = {k: " ".join(set(v)) for k, v in right_entry.items()}
        right = right.append(
            pd.DataFrame(right_entry, index=[desc["subject_id"]])
        )

        instance = (dataset).lower()
        judgement = row['results'][f"does_the_{instance}_description_sufficiently_describe_the_same_given_{instance}"]

        gt_label = judgement["agg"]
        gt_confidence = judgement["confidence"]

        if len(right_entry) == 0:
            continue

        gt = gt.append(
            pd.DataFrame({
                "id_l": wikidata_ent,
                "id_r": desc["subject_id"],
                "match": gt_label,
                "confidence": gt_confidence
            }, index=[0])
        )

    left.drop_duplicates(inplace=True)
    right.drop_duplicates(inplace=True)
    gt = gt.drop_duplicates().reset_index(drop=True)
    gt = gt[gt['match'].isin(["yes", "insufficient_info"])].drop(
        ['match', 'confidence'], axis=1).reset_index(drop=True)

    right = right.reset_index().rename(
        columns={'index': 'id', 'name': 'title'})
    left = left.reset_index().rename(columns={'index': 'id'})

    left = left.drop(left.filter(regex='ID').columns, axis=1)
    right = right.drop(right.filter(regex='ID').columns, axis=1)

    mapping = align_attributes(datadir, dataset, variant)

    leftA = left[mapping.keys()].rename(mapping, axis=1)
    rightA = right[mapping.values()]

    gtA = gt
    gtA = pd.merge(gtA, leftA.add_suffix('_l'), how="inner", on="id_l")
    gtA = pd.merge(gtA, rightA.add_suffix('_r'), how="inner", on="id_r")

    if variant == "full":
        leftA.drop_duplicates().dropna(how="all", subset=[
            c for c in leftA.columns if "id" not in c]).to_csv(subset_mapping["left"], index=False)
        rightA.drop_duplicates().dropna(how="all", subset=[
            c for c in rightA.columns if "id" not in c]).to_csv(subset_mapping["right"], index=False)
        gtA.drop_duplicates().dropna(how="all", subset=[
            c for c in gtA.columns if "id" not in c]).to_csv(subset_mapping["gt"], index=False)
    else:
        cols = None
        if variant == "1col":
            cols = ["id", "title"]
        elif variant == "2col":
            if dataset == "Book":
                cols = ["id", "title", "author"]
            elif dataset == "Movie":
                cols = ["id", "title", "director"]

        leftA[cols].drop_duplicates().dropna().to_csv(
            subset_mapping["left"], index=False)
        rightA[cols].drop_duplicates().dropna().to_csv(
            subset_mapping["right"], index=False)
        gtA[list(map(lambda x: x + "_l", cols)) + list(map(lambda x: x + "_r", cols))
            ].dropna().drop_duplicates().to_csv(subset_mapping["gt"], index=False)


@cli.command()
@click.argument("datadir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dataset", type=click.STRING)
@click.argument("variant", type=click.STRING)
def augment_left(datadir, dataset, variant):
    # Augment left dataset
    left_aug = pd.DataFrame()
    # pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    # processor = spacy.load("en_core_web_lg")
    # homedir = os.path.join(datadir, f"KM_{dataset}_{variant}")
    # left = pd.read_csv(os.path.join(homedir, "left.csv"))
    # for block, title in tqdm(enumerate(left["title"]), total=left["title"].count()):
    #     keywords = []
    #     doc = processor(title.lower())
    #     for token in doc:
    #         if(token.text in processor.Defaults.stop_words or token.text in punctuation):
    #             continue

    #         if(token.pos_ in pos_tag):
    #             keywords.append(token.text)
    #         else:
    #             print(token.pos_, token.text)

    wd_aug_ents = get_wikidata_ent_sparql(dataset)
    for entId, entAux in wd_aug_ents.items():
        # Augmentation
        title = get_property_label(entId)
        entAux["title (en)"] = title
        entry = pd.DataFrame(entAux, index=[entId])
        #entry["block"] = block
        left_aug = left_aug.append(entry)

    mapping = align_attributes(datadir, dataset, variant)

    (left_aug
        .reset_index()
        .rename({"index": "id"}, axis=1)[list(mapping.keys())]
        .rename(mapping, axis=1)
        .drop_duplicates()
        .dropna(how="all", subset=["title"])
        .to_csv("left_aug.csv", index=False)
     )


if __name__ == '__main__':
    cli()
