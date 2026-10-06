"""Print the primary point F1 (tau 2%) of results files.
Usage: python f1.py results/<file>.json [...]"""

import json
import sys

for f in sys.argv[1:]:
    pm = json.load(open(f))["point_metrics"]
    row = pm["by_tau"]
    row = row.get(str(pm["primary_tau"])) or row.get(f"{pm['primary_tau']}") or row
    print(f, json.dumps(row)[:400])
