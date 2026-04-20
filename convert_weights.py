"""
Convert PVFinder .pyt weights to binary format for the Allen C++ implementation.

Produces two files:
  fc_weights.bin  — FC MLP layers 1–6A (used by PVFinderTrackAggregation)
  cnn_weights.bin — UNet CNN layers     (used by PVFinderUNet)

Usage:
  python convert_weights.py --model <path/to/model.pyt> \\
      [--fc-out fc_weights.bin] [--cnn-out cnn_weights.bin]

The script auto-detects the UNet feature count (N_FEAT) from the state dict.
The FC architecture (9→20→20→20→20→20→800) is the same across all variants.

NOTE: When using a 16-channel model, Allen must be rebuilt with N_FEAT=16 in
PVFinderUNet.cuh before the produced cnn_weights.bin will work.
"""

import argparse
import sys
import struct
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers shared by FC and CNN converters
# ---------------------------------------------------------------------------

def _w32(f, arr):
    f.write(arr.astype(np.float32).tobytes())

def _i32(f, val):
    f.write(struct.pack('<i', int(val)))


# ---------------------------------------------------------------------------
# FC converter  (layers 1-6A, layout identical regardless of UNet width)
# ---------------------------------------------------------------------------

FC_LAYOUT = [
    ("layer1",  9,  None),   # 9→nOut1
    ("layer2",  None, None), # nOut1→nOut2
    ("layer3",  None, None),
    ("layer4",  None, None),
    ("layer5",  None, None),
    ("layer6A", None, None), # nOut5 → latentChannels*100
]

def write_fc_weights(f, state_dict):
    """Write FC weight binary (flat floats, no header).

    Layout per layer: W (out×in) then b (out), row-major.
    layer6A W is transposed to [in×out] for coalesced GEMM access in Allen.
    """
    layer_names = ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6A"]
    for name in layer_names:
        w = state_dict[f"{name}.weight"].cpu().numpy().astype(np.float32)
        b = state_dict[f"{name}.bias"].cpu().numpy().astype(np.float32)
        out_c, in_c = w.shape
        print(f"  FC {name}: [{in_c} → {out_c}]")
        if name == "layer6A":
            # Transpose: Allen kernel expects [in_c, out_c] for coalesced access
            w = w.T.copy()
        f.write(w.tobytes())
        f.write(b.tobytes())


# ---------------------------------------------------------------------------
# CNN converter  (UNet, arbitrary N_FEAT)
# ---------------------------------------------------------------------------

def _write_conv1d(f, key_w, key_b, state_dict, label):
    w = state_dict[key_w].cpu().numpy().astype(np.float32)  # [out, in, k]
    b = state_dict[key_b].cpu().numpy().astype(np.float32)  # [out]
    out_c, in_c, k = w.shape
    print(f"  CNN {label}: Conv1d({in_c}→{out_c}, k={k})")
    _i32(f, in_c); _i32(f, out_c); _i32(f, k)
    _w32(f, w); _w32(f, b)

def _write_bn1d(f, prefix, state_dict, label):
    gamma = state_dict[f"{prefix}.weight"].cpu().numpy().astype(np.float32)
    beta  = state_dict[f"{prefix}.bias"].cpu().numpy().astype(np.float32)
    mean  = state_dict[f"{prefix}.running_mean"].cpu().numpy().astype(np.float32)
    var   = state_dict[f"{prefix}.running_var"].cpu().numpy().astype(np.float32)
    eps   = float(1e-5)  # PyTorch default
    n     = len(gamma)
    print(f"  CNN {label}: BN1d(features={n})")
    _i32(f, n)
    f.write(struct.pack('<f', eps))
    _w32(f, gamma); _w32(f, beta); _w32(f, mean); _w32(f, var)

def _write_convt1d(f, key_w, key_b, state_dict, label):
    w = state_dict[key_w].cpu().numpy().astype(np.float32)  # [in, out, k]
    b = state_dict[key_b].cpu().numpy().astype(np.float32)  # [out]
    in_c, out_c, k = w.shape
    stride = 2
    print(f"  CNN {label}: ConvTranspose1d({in_c}→{out_c}, k={k}, s={stride})")
    _i32(f, in_c); _i32(f, out_c); _i32(f, k); _i32(f, stride)
    _w32(f, w); _w32(f, b)

def write_cnn_weights(f, state_dict):
    """Write CNN weight binary with magic 0xCAFE0001."""
    f.write(struct.pack('<I', 0xCAFE0001))

    # rcbn1: Conv(N_BATCH_CHANNELS→N_FEAT, k=25) + BN
    _write_conv1d(f, "rcbn1.0.weight", "rcbn1.0.bias", state_dict, "rcbn1.conv")
    _write_bn1d(f, "rcbn1.1", state_dict, "rcbn1.bn")

    # rcbn2: Conv(N_FEAT→N_FEAT, k=7) + BN
    _write_conv1d(f, "rcbn2.0.weight", "rcbn2.0.bias", state_dict, "rcbn2.conv")
    _write_bn1d(f, "rcbn2.1", state_dict, "rcbn2.bn")

    # rcbn3: Conv(N_FEAT→N_FEAT, k=5) + BN
    _write_conv1d(f, "rcbn3.0.weight", "rcbn3.0.bias", state_dict, "rcbn3.conv")
    _write_bn1d(f, "rcbn3.1", state_dict, "rcbn3.bn")

    # up1: ConvTranspose(N_FEAT→N_FEAT, k=2, s=2) + Conv(N_FEAT→N_FEAT, k=5) + BN
    _write_convt1d(f, "up1.0.weight", "up1.0.bias", state_dict, "up1.convt")
    _write_conv1d(f, "up1.1.0.weight", "up1.1.0.bias", state_dict, "up1.conv")
    _write_bn1d(f, "up1.1.1", state_dict, "up1.bn")

    # up2: ConvTranspose(2*N_FEAT→N_FEAT, k=2, s=2) + Conv(N_FEAT→N_FEAT, k=5) + BN
    _write_convt1d(f, "up2.0.weight", "up2.0.bias", state_dict, "up2.convt")
    _write_conv1d(f, "up2.1.0.weight", "up2.1.0.bias", state_dict, "up2.conv")
    _write_bn1d(f, "up2.1.1", state_dict, "up2.bn")

    # out_intermediate: Conv(2*N_FEAT→N_FEAT, k=5)
    _write_conv1d(f, "out_intermediate.weight", "out_intermediate.bias", state_dict, "out_intermediate")

    # outc: Conv(N_FEAT→1, k=5)
    _write_conv1d(f, "outc.weight", "outc.bias", state_dict, "outc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_n_feat(state_dict):
    """Infer N_FEAT from the first conv layer weight shape."""
    w = state_dict["rcbn1.0.weight"]
    return w.shape[0]  # out_channels of rcbn1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to .pyt state dict")
    p.add_argument("--fc-out",  default="fc_weights.bin",  help="Output FC binary [fc_weights.bin]")
    p.add_argument("--cnn-out", default="cnn_weights.bin", help="Output CNN binary [cnn_weights.bin]")
    p.add_argument("--fc-only",  action="store_true", help="Only write FC weights")
    p.add_argument("--cnn-only", action="store_true", help="Only write CNN weights")
    args = p.parse_args()

    print(f"Loading model from: {args.model}")
    state_dict = torch.load(args.model, map_location="cpu")
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    n_feat = detect_n_feat(state_dict)
    n_latent = state_dict["layer6A.bias"].shape[0] // 100  # latentChannels
    print(f"Detected: N_FEAT={n_feat}, latentChannels={n_latent}")

    if n_feat != 64:
        print(f"\nWARNING: N_FEAT={n_feat} detected. Allen must be rebuilt with")
        print(f"  N_FEAT = {n_feat}  in Allen/device/pvfinder/include/PVFinderUNet.cuh")
        print(f"  before cnn_weights.bin will work correctly.\n")

    if not args.cnn_only:
        print(f"\nWriting FC weights → {args.fc_out}")
        with open(args.fc_out, "wb") as f:
            write_fc_weights(f, state_dict)
        size = sum(state_dict[k].numel() for k in state_dict if k.startswith("layer"))
        print(f"  Total FC floats: {size:,}  ({size*4/1024:.1f} KB)")

    if not args.fc_only:
        print(f"\nWriting CNN weights → {args.cnn_out}")
        with open(args.cnn_out, "wb") as f:
            write_cnn_weights(f, state_dict)
        import os
        sz = os.path.getsize(args.cnn_out)
        print(f"  File size: {sz/1024:.1f} KB")

    print("\nDone.")


if __name__ == "__main__":
    main()
