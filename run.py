"""Generate the WHC submission file from the official tactical data.

    python run.py --data-dir data --out WHC.csv

Reads only local files: the tactical ATD (.txt + .xml), the Smart Tagging XML and config/whc.yaml.
Same logic as the --tactical path of the development runner; no answers are read.
"""

import argparse
import os

import pandas as pd
import yaml

from algorithms.dead_ball_aware.predict import predict
from observations.atd import load_tactical

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config", "whc.yaml")


def load_tags(data_dir=None):
    """Adapt official Smart Tagging intervals to the algorithm's tag records."""
    from starter.load_smart_tagging import load

    teams = {'Barcelona': 'H', 'Atletic Club': 'A', 'N/A': None}
    return [{'start_frame': event['start_frame'], 'end_frame': event['end_frame'],
             'code': event['code'], 'type': event['tags'].get('Type'),
             'team': teams[event['tags']['Team']]}
            for event in load(data_dir=data_dir)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data", help="folder that contains tactical/")
    parser.add_argument("--out", default="WHC.csv", help="output frame,label CSV")
    args = parser.parse_args()

    with open(CONFIG) as handle:
        params = yaml.safe_load(handle)
    tags = load_tags(args.data_dir)
    observations, metadata = load_tactical(args.data_dir)
    labels = list(predict(observations, metadata, params, tags=tags))
    if len(labels) != len(observations):
        raise ValueError("{} labels for {} frames".format(len(labels), len(observations)))
    result = pd.DataFrame({"frame": observations["frame"].to_numpy(), "label": labels})
    result.to_csv(args.out, index=False)
    print("wrote {} ({} rows)".format(args.out, len(result)))


if __name__ == "__main__":
    main()
