#!/usr/bin/env python3
"""
Generate deterministic fake PVFinder FC weights for Allen performance tests.

The binary layout matches PVFinderFCAggregation.cu before its runtime
final-layer transpose:

  hidden layer 1: W[out, in] then b[out]
  hidden layers 2..N: W[out, hidden] then b[out]
  final layer: W[out, hidden] then b[out]

These weights are intentionally meaningless for physics. They only let the
Allen FC kernels run with the correct tensor shapes for timing studies.
"""

import argparse
import random
import struct
from pathlib import Path


def add_layer(parts, rng, in_dim, out_dim, scale):
    weight = [rng.gauss(0.0, scale) for _ in range(out_dim * in_dim)]
    bias = [rng.gauss(0.0, scale) for _ in range(out_dim)]
    parts.extend([weight, bias])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="fc_weights_32x3_400.bin")
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--output-dim", type=int, default=400)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--scale", type=float, default=0.02)
    args = parser.parse_args()

    if args.hidden_layers < 1:
        raise SystemExit("--hidden-layers must be at least 1")
    if args.output_dim % 100 != 0:
        raise SystemExit("--output-dim must be latentChannels * 100")

    rng = random.Random(args.seed)
    parts = []

    add_layer(parts, rng, args.input_dim, args.hidden_dim, args.scale)
    for _ in range(1, args.hidden_layers):
        add_layer(parts, rng, args.hidden_dim, args.hidden_dim, args.scale)
    add_layer(parts, rng, args.hidden_dim, args.output_dim, args.scale)

    out = Path(args.out)
    with out.open("wb") as handle:
        for part in parts:
            handle.write(struct.pack(f"<{len(part)}f", *part))

    n_floats = sum(len(part) for part in parts)
    print(f"Wrote {out} ({n_floats} float32 values, {n_floats * 4} bytes)")
    print(
        "Architecture: "
        f"{args.input_dim}->{args.hidden_dim}x{args.hidden_layers}->{args.output_dim}"
    )


if __name__ == "__main__":
    main()
