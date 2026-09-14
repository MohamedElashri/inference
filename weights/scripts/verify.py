#!/usr/bin/env python3
"""verify.py - Check Allen weight files against the checkpoint they came from.

Reads fc_weights.bin and cnn_weights.bin independently of convert.py, in the
order and layout Allen's loaders read them (PVFinderFCAggregation.cu and
load_weights in PVFinderUNet.cu), and compares every tensor bit for bit with
the checkpoint (as float32). Also reports the Allen build this model needs and
whether Allen's UNet loader supports its skip-connection mode.

Usage (normally through the pipeline: make -C weights verify MODEL=<name>):
    python3 weights/scripts/verify.py --checkpoint MODEL.pyt --fc fc_weights.bin --cnn cnn_weights.bin

Exit status 0 only if both files match the checkpoint exactly.
"""
import argparse
import struct
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser(description="Verify Allen weight files against a checkpoint")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--fc", required=True, help="fc_weights.bin")
parser.add_argument("--cnn", required=True, help="cnn_weights.bin")
args = parser.parse_args()

sd = torch.load(args.checkpoint, map_location="cpu")
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()


def tensor(key):
    return sd[key].float().cpu().numpy()


problems = []


def check(label, got, want):
    if got.shape != want.shape or not np.array_equal(got, want):
        problems.append(label)
        print(f"  MISMATCH {label}: file {got.shape} vs checkpoint {want.shape}")


# ---------------------------------------------------------------------------
# FC: flat float32, per layer W (out x in, row-major) then b; layer6A included.
# ---------------------------------------------------------------------------
fc = np.fromfile(args.fc, dtype=np.float32)
off = 0
for k in ("1", "2", "3", "4", "5", "6A"):
    w, b = tensor(f"layer{k}.weight"), tensor(f"layer{k}.bias")
    if off + w.size + b.size > fc.size:
        problems.append(f"fc layer{k}: file too short")
        break
    check(f"fc layer{k}.weight", fc[off:off + w.size].reshape(w.shape), w)
    off += w.size
    check(f"fc layer{k}.bias", fc[off:off + b.size], b)
    off += b.size
if off != fc.size:
    problems.append(f"fc: {fc.size - off} trailing floats")
latent = tensor("layer6A.bias").size // 100
print(f"fc_weights.bin : {fc.size} floats, latentChannels={latent}, "
      f"hidden={[tensor(f'layer{k}.weight').shape[0] for k in '12345']}")


# ---------------------------------------------------------------------------
# CNN: magic 0xCAFE0001 then, in load_weights order,
#   conv   int32 in, out, k   | float32 W[out,in,k] | b[out]
#   bn     int32 n | float32 eps | gamma, beta, mean, var [n]
#   convT  int32 in, out, k, stride | float32 W[in,out,k] | b[out]
# ---------------------------------------------------------------------------
class Reader:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.pos = 0

    def ints(self, n):
        vals = struct.unpack_from(f"<{n}i", self.buf, self.pos)
        self.pos += 4 * n
        return vals

    def floats(self, n):
        arr = np.frombuffer(self.buf, dtype=np.float32, count=n, offset=self.pos)
        self.pos += 4 * n
        return arr


cnn = Reader(args.cnn)
(magic,) = struct.unpack_from("<I", cnn.buf, 0)
cnn.pos = 4
if magic != 0xCAFE0001:
    sys.exit(f"cnn_weights.bin: bad magic {magic:#x}")


def conv(prefix):
    cin, cout, k = cnn.ints(3)
    check(f"cnn {prefix}.weight", cnn.floats(cout * cin * k).reshape(cout, cin, k), tensor(f"{prefix}.weight"))
    check(f"cnn {prefix}.bias", cnn.floats(cout), tensor(f"{prefix}.bias"))
    return cin, cout


def bn(prefix):
    (n,) = cnn.ints(1)
    eps = cnn.floats(1)[0]
    if not np.isclose(eps, 1e-5):
        problems.append(f"cnn {prefix}: eps {eps} != 1e-5")
    for name, key in (("gamma", "weight"), ("beta", "bias"), ("mean", "running_mean"), ("var", "running_var")):
        check(f"cnn {prefix}.{name}", cnn.floats(n), tensor(f"{prefix}.{key}"))


def convt(prefix):
    cin, cout, k, stride = cnn.ints(4)
    if stride != 2:
        problems.append(f"cnn {prefix}: stride {stride} != 2")
    check(f"cnn {prefix}.weight", cnn.floats(cin * cout * k).reshape(cin, cout, k), tensor(f"{prefix}.weight"))
    check(f"cnn {prefix}.bias", cnn.floats(cout), tensor(f"{prefix}.bias"))


try:
    in_ch, n_feat = conv("rcbn1.0"); bn("rcbn1.1")
    conv("rcbn2.0"); bn("rcbn2.1")
    conv("rcbn3.0"); bn("rcbn3.1")
    convt("up1.0"); conv("up1.1.0"); bn("up1.1.1")
    convt("up2.0"); conv("up2.1.0"); bn("up2.1.1")
    oint_in, _ = conv("out_intermediate")
    conv("outc")
except (struct.error, ValueError) as exc:
    sys.exit(f"cnn_weights.bin: truncated or malformed ({exc})")
if cnn.pos != len(cnn.buf):
    problems.append(f"cnn: {len(cnn.buf) - cnn.pos} trailing bytes")
if in_ch != latent:
    problems.append(f"cnn input channels {in_ch} != FC latentChannels {latent}")

concat = oint_in == 2 * n_feat
print(f"cnn_weights.bin: N_FEAT={n_feat}, input channels={in_ch}, "
      f"skip connections={'concat' if concat else 'add/none'}")
print(f"Allen build flags: --unet-feat {n_feat} --unet-batch-channels {latent}")
print(f"Allen UNet loader: {'supported' if concat else 'NOT supported (needs concatenated skip connections); FC stage only'}")

if problems:
    print(f"FAIL: {len(problems)} problem(s): {', '.join(problems[:6])}{' ...' if len(problems) > 6 else ''}")
    sys.exit(1)
print(f"OK: both files match {args.checkpoint} bit for bit")
