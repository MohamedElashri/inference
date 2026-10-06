#!/usr/bin/env python3
"""Restore a measured Allen configuration from its tracked object delta."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("delta", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
delta = json.loads(args.delta.read_text())
if delta["schema"] != "allen-config-delta/1":
    parser.error("unsupported configuration delta schema")
config = json.loads((args.delta.parent / delta["canonical"]).read_text())
for key in delta["removed_keys"]:
    config.pop(key)
config.update(delta["overrides"])
args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
