"""
Convert a PVFinder .pyt checkpoint to the model file Allen reads.

Produces one Allen tensor model file (format "allen-tensors/1", kind
"pvfinder"), read by the Allen algorithms pvfinder_fc_aggregation and
pvfinder_unet through their "model" property (see
Allen/device/utils/mva_models/include/TensorModel.h and
Allen/device/pvfinder/include/PVFinderModel.h):

  {"format": "allen-tensors/1", "kind": "pvfinder", "name": ...,
   "source": ..., "sha256": ...,
   "metadata": {"latent_channels": 4, "unet_features": 16, "bn_eps": 1e-05},
   "tensors": {"layer1.weight": {"shape": [20, 9], "data": [...]}, ...}}

Tensors keep their PyTorch state-dict names and shapes, data row major; each
value is written as the shortest decimal that reads back as the same float32,
so the file is exact.

Usage (normally through the pipeline: make -C weights convert MODEL=<name>):
  python weights/scripts/convert.py --model <path/to/model.pyt> --out pvfinder_model.json [--name NAME]

The UNet has no skip connections; a checkpoint trained with them is rejected.
Allen must be built to match the model's widths, e.g. for the 16-feature,
latentChannels-4 model:
  PVFINDER_UNET_N_FEAT=16 PVFINDER_UNET_N_BATCH_CHANNELS=4 bash benchmarks/build_allen.sh
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

# The tensors Allen reads, by state-dict name.
FC_TENSORS = [f"{layer}.{kind}" for layer in ("layer1", "layer2", "layer3", "layer4", "layer5", "layer6A")
              for kind in ("weight", "bias")]
BN = ("weight", "bias", "running_mean", "running_var")
UNET_TENSORS = (
    ["rcbn1.0.weight", "rcbn1.0.bias"] + [f"rcbn1.1.{k}" for k in BN]
    + ["rcbn2.0.weight", "rcbn2.0.bias"] + [f"rcbn2.1.{k}" for k in BN]
    + ["rcbn3.0.weight", "rcbn3.0.bias"] + [f"rcbn3.1.{k}" for k in BN]
    + ["up1.0.weight", "up1.0.bias", "up1.1.0.weight", "up1.1.0.bias"] + [f"up1.1.1.{k}" for k in BN]
    + ["up2.0.weight", "up2.0.bias", "up2.1.0.weight", "up2.1.0.bias"] + [f"up2.1.1.{k}" for k in BN]
    + ["out_intermediate.weight", "out_intermediate.bias", "outc.weight", "outc.bias"]
)
BN_EPS = 1e-5  # PyTorch's BatchNorm1d default, as in training
FORMAT, KIND = "allen-tensors/1", "pvfinder"


def float32_list(array):
    """Values as Python floats; json writes the shortest repr, which reads back as the same float32."""
    return [float(v) for v in np.asarray(array, dtype=np.float32).ravel()]


def load_state_dict(path):
    sd = torch.load(path, map_location="cpu")
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    return sd


def model_json(state_dict, name, source):
    n_feat = int(state_dict["rcbn1.0.weight"].shape[0])
    n_latent = int(state_dict["layer6A.bias"].shape[0]) // 100
    up2_in = int(state_dict["up2.0.weight"].shape[0])              # ConvTranspose1d: [in, out, k]
    oint_in = int(state_dict["out_intermediate.weight"].shape[1])  # Conv1d: [out, in, k]
    if up2_in != n_feat or oint_in != n_feat:
        sys.exit(f"ERROR: up2 takes {up2_in} and out_intermediate {oint_in} input channels, expected "
                 f"{n_feat}: this checkpoint has skip connections, which Allen's UNet does not implement")
    tensors = {}
    for key in FC_TENSORS + UNET_TENSORS:
        t = state_dict[key].float().cpu().numpy()
        tensors[key] = {"shape": list(t.shape), "data": float32_list(t)}
    sha = hashlib.sha256(open(source, "rb").read()).hexdigest()
    return {
        "format": FORMAT,
        "kind": KIND,
        "name": name,
        "source": os.path.abspath(source),
        "sha256": sha,
        "metadata": {"latent_channels": n_latent, "unet_features": n_feat, "bn_eps": BN_EPS},
        "tensors": tensors,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to the .pyt state dict")
    p.add_argument("--out", default="pvfinder_model.json", help="output model file [pvfinder_model.json]")
    p.add_argument("--name", default=None, help="model name recorded in the file [checkpoint file name]")
    args = p.parse_args()

    print(f"Loading {args.model}")
    state_dict = load_state_dict(args.model)
    name = args.name or os.path.splitext(os.path.basename(args.model))[0]
    model = model_json(state_dict, name, args.model)
    with open(args.out, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    n = sum(len(t["data"]) for t in model["tensors"].values())
    print(f"Wrote {args.out}: {len(model['tensors'])} tensors, {n:,} floats, "
          f"N_FEAT={model['metadata']['unet_features']}, latentChannels={model['metadata']['latent_channels']}")
    print(f"Allen build for this model: PVFINDER_UNET_N_FEAT={model['metadata']['unet_features']} "
          f"PVFINDER_UNET_N_BATCH_CHANNELS={model['metadata']['latent_channels']} bash benchmarks/build_allen.sh")


if __name__ == "__main__":
    main()
