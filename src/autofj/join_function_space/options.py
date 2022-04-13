"""Options of join functions"""

import yaml
import os
import pkgutil

config = yaml.safe_load(pkgutil.get_data(__name__, "autofj_jfs.yml"))

autofj_lg = config["autofj_lg"]
autofj_md = config["autofj_md"]
autofj_sm = config["autofj_sm"]