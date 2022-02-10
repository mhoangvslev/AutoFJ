"""Options of join functions"""

import yaml
import os

config = yaml.safe_load(open(os.path.join("config", "autofj.yml"), mode="rb"))

autofj_lg = config["autofj_lg"]
autofj_md = config["autofj_md"]
autofj_sm = config["autofj_sm"]