#!/usr/bin/env python3
"""verify.py - Check an Allen PVFinder model file against the checkpoint it came from.

Reads the model file (pvfinder_model.json, format allen-tensors/1, kind pvfinder) as Allen
does, independently of convert.py: every tensor it holds, parsed and rounded
to float32, must equal the checkpoint's bit for bit, with the checkpoint's
shape. Also checks the metadata (latent channels, UNet features, no skip
connections, BatchNorm epsilon) and reports the Allen build the model needs.

Usage (normally through the pipeline: make -C weights verify MODEL=<name>):
    python3 weights/scripts/verify.py --checkpoint MODEL.pyt --model-file pvfinder_model.json

Exit status 0 only if the file matches the checkpoint exactly.
"""
import argparse
import json
import sys

import numpy as np
import torch

from convert import BN_EPS, FC_TENSORS, FORMAT, KIND, UNET_TENSORS

parser = argparse.ArgumentParser(description="Verify an Allen PVFinder model file against a checkpoint")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--model-file", required=True, help="pvfinder_model.json")
args = parser.parse_args()

sd = torch.load(args.checkpoint, map_location="cpu")
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()
with open(args.model_file) as fp:
    model = json.load(fp)

problems = []
if model.get("format") != FORMAT:
    problems.append(f"format is {model.get('format')!r}, not {FORMAT!r}")
if model.get("kind") != KIND:
    problems.append(f"kind is {model.get('kind')!r}, not {KIND!r}")
meta = model.get("metadata", {})
n_feat = int(sd["rcbn1.0.weight"].shape[0])
n_latent = int(sd["layer6A.bias"].shape[0]) // 100
if meta.get("unet_features") != n_feat:
    problems.append(f"unet_features {meta.get('unet_features')} != checkpoint {n_feat}")
if meta.get("latent_channels") != n_latent:
    problems.append(f"latent_channels {meta.get('latent_channels')} != checkpoint {n_latent}")
if np.float32(meta.get("bn_eps", 0.0)) != np.float32(BN_EPS):
    problems.append(f"bn_eps {meta.get('bn_eps')} != {BN_EPS}")
if int(sd["up2.0.weight"].shape[0]) != n_feat or int(sd["out_intermediate.weight"].shape[1]) != n_feat:
    problems.append("the checkpoint has skip connections, which Allen's UNet does not implement")

tensors = model.get("tensors", {})
for key in FC_TENSORS + UNET_TENSORS:
    want = sd[key].float().cpu().numpy()
    if key not in tensors:
        problems.append(f"{key}: missing")
        continue
    t = tensors[key]
    if list(t["shape"]) != list(want.shape):
        problems.append(f"{key}: shape {t['shape']} != checkpoint {list(want.shape)}")
        continue
    got = np.asarray(t["data"], dtype=np.float64).astype(np.float32).reshape(want.shape)
    if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
        n_bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
        problems.append(f"{key}: {n_bad} of {want.size} values differ from the checkpoint")
extra = sorted(set(tensors) - set(FC_TENSORS + UNET_TENSORS))
if extra:
    problems.append(f"unexpected tensors: {', '.join(extra)}")

n_values = sum(len(t["data"]) for t in tensors.values())
print(f"{args.model_file}: {len(tensors)} tensors, {n_values:,} floats; N_FEAT={n_feat}, latentChannels={n_latent}")
print(f"Allen build: ./ballen -a gpu --cudnn --unet-feat {n_feat} --unet-batch-channels {n_latent}")
if problems:
    for p in problems:
        print(f"  MISMATCH {p}")
    print("FAIL")
    sys.exit(1)
print("every tensor equals the checkpoint bit for bit: PASS")
