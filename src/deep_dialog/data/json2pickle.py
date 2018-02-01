#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import pickle
import sys

import better_exceptions

def load_json(path):
    with open(path, "r") as f:
        j = json.load(f)
    return j


def save_pickle(d, path):
    with open(path, "wb") as f:
        pickle.dump(d, f)


def main():
    # 引数を解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--src")
    parser.add_argument("--out")
    args = parser.parse_args()

    # 引数を変数に設定
    src_file_path = args.src
    out_file_path = args.out

    d = load_json(src_file_path)

    save_pickle(d, out_file_path)


if __name__ == "__main__":
    print(sys.argv[1])
    d = load_json(sys.argv[1])
    print(sys.argv[2])
    save_pickle(d, sys.argv[2])
    # main()
