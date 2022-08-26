import preprocessing.loader as lod
from pathlib import Path
import logging
import sys
import json

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

dataset = lod.build_dataset(Path(__file__).parent / "datasets/train", "en")

json_object = json.dumps(dataset, indent=4)
with open("dataset.json", 'w') as fp:
    fp.write(json_object)

