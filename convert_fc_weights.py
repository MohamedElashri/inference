import sys

import numpy as np
import torch


def write_linear(f, layer, name):
    print(
        f"Extracting {name} "
        f"(Linear: in={layer.in_features}, out={layer.out_features})..."
    )
    w = layer.weight.detach().cpu().numpy().astype(np.float32)
    b = layer.bias.detach().cpu().numpy().astype(np.float32)
    f.write(w.tobytes())
    f.write(b.tobytes())


def main():
    sys.path.append("pvfinder_pytorch")
    from utils import TrackIntervalsToKDE_HDplusUNet100 as Model

    model = Model()
    weight_path = (
        "pvfinder_pytorch/weights/"
        "07Sept2023_t2hists_HDplusUNet100_iter12Ca_200epochs_2em5_5p0_final.pyt"
    )
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    output_file = "fc_weights.bin"
    print(f"Writing FC weights to {output_file}...")

    with open(output_file, "wb") as f:
        write_linear(f, model.layer1, "layer1")
        write_linear(f, model.layer2, "layer2")
        write_linear(f, model.layer3, "layer3")
        write_linear(f, model.layer4, "layer4")
        write_linear(f, model.layer5, "layer5")
        write_linear(f, model.layer6A, "layer6A")

    print(f"Successfully wrote {output_file}.")


if __name__ == "__main__":
    main()
