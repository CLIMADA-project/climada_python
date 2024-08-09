import json
import sys

key = sys.argv[1]
val = sys.argv[2]
jsonfile = "climada.conf"

with open(jsonfile, encoding="UTF-8") as inf:
    data = json.load(inf)
data[key] = val
with open(jsonfile, "w", encoding="UTF-8") as outf:
    json.dump(data, outf)
