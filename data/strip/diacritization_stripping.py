#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import argparse
import sys

import diacritization_stripping_data

parser = argparse.ArgumentParser()
parser.add_argument("--uninorms", action="store_true",
                    help="Use diacritization stripping based on Unicode Normalization")
parser.add_argument("--uninames", action="store_true",
                    help="Use diacritization stripping based on Unicode Names")
args = parser.parse_args()

maps = []
if args.uninames: maps.append(diacritization_stripping_data.strip_diacritization_uninames)
if args.uninorms: maps.append(diacritization_stripping_data.strip_diacritization_uninorms)

total, stripped, stripped_map = 0, 0, {}
for line in sys.stdin:
    output = ""
    for c in line:
        for m in maps:
            if c in m:
                stripped += 1
                stripped_map[c] = stripped_map.get(c, 0) + 1
                output += m[c]
                break
        else:
            output += c

        if not c.isspace():
            total += 1
    print(output, end="")