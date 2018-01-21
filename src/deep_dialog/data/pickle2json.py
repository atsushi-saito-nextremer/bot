#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import pickle
import sys


def load_pickle(path):
    try:
        with open(path, "rb") as f:
            p = pickle.load(f)
    except EOFError:
        return {}
    return np2list(p)


def np2list(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            for key2 in d[key].keys():
                if type(d[key][key2]).__module__ == "numpy":
                    d[key][key2] = []

    return d


def save_json(jdict, path):
    with open(path, "w") as f:
        json.dump(jdict, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    d = load_pickle(sys.argv[1])

    save_json(d, sys.argv[2])
