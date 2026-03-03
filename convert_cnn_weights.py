import torch
import numpy as np
import sys

def write_conv1d(f, layer, name):
    print(f"Extracting {name} (Conv1d: in={layer.in_channels}, out={layer.out_channels}, k={layer.kernel_size[0]})...")
    w = layer.weight.detach().cpu().numpy().astype(np.float32) # [out_c, in_c, k]
    b = layer.bias.detach().cpu().numpy().astype(np.float32)   # [out_c]
    
    # Write metadata
    f.write(np.int32(layer.in_channels).tobytes())
    f.write(np.int32(layer.out_channels).tobytes())
    f.write(np.int32(layer.kernel_size[0]).tobytes())
    
    # Write data
    f.write(w.tobytes())
    f.write(b.tobytes())

def write_batchnorm1d(f, layer, name):
    print(f"Extracting {name} (BatchNorm1d: features={layer.num_features})...")
    w = layer.weight.detach().cpu().numpy().astype(np.float32)
    b = layer.bias.detach().cpu().numpy().astype(np.float32)
    rm = layer.running_mean.detach().cpu().numpy().astype(np.float32)
    rv = layer.running_var.detach().cpu().numpy().astype(np.float32)
    eps = np.float32(layer.eps)
    
    # Write metadata
    f.write(np.int32(layer.num_features).tobytes())
    f.write(eps.tobytes())
    
    # Write data
    f.write(w.tobytes())
    f.write(b.tobytes())
    f.write(rm.tobytes())
    f.write(rv.tobytes())

def write_convbnrelu(f, seq, name):
    # seq is nn.Sequential(Conv1d, BatchNorm1d, ReLU, Dropout)
    write_conv1d(f, seq[0], f"{name}.conv")
    write_batchnorm1d(f, seq[1], f"{name}.bn")

def write_up(f, seq, name):
    # seq is nn.Sequential(ConvTranspose1d, ConvBNrelu)
    
    # Extract ConvTranspose1d
    conv_t = seq[0]
    print(f"Extracting {name}.convT (ConvTranspose1d: in={conv_t.in_channels}, out={conv_t.out_channels}, k={conv_t.kernel_size[0]}, stride={conv_t.stride[0]})...")
    # cuDNN for ConvTranspose1d expects filter in shape [in_c, out_c, k] natively?
    # PyTorch stores it as [in_c, out_c, k]. We just dump it out.
    w = conv_t.weight.detach().cpu().numpy().astype(np.float32)
    b = conv_t.bias.detach().cpu().numpy().astype(np.float32)
    
    f.write(np.int32(conv_t.in_channels).tobytes())
    f.write(np.int32(conv_t.out_channels).tobytes())
    f.write(np.int32(conv_t.kernel_size[0]).tobytes())
    f.write(np.int32(conv_t.stride[0]).tobytes())
    
    f.write(w.tobytes())
    f.write(b.tobytes())

    # Extract ConvBNrelu
    write_convbnrelu(f, seq[1], f"{name}.convbnrelu")


def main():
    sys.path.append('pvfinder_pytorch')
    from utils import TrackIntervalsToKDE_HDplusUNet100 as Model
    
    model = Model()
    name = 'pvfinder_pytorch/weights/07Sept2023_t2hists_HDplusUNet100_iter12Ca_200epochs_2em5_5p0_final.pyt'
    d = torch.load(name, map_location='cpu')
    model.load_state_dict(d)
    model.eval()

    output_file = 'cnn_weights.bin'
    print(f"Writing CNN weights to {output_file}...")
    
    with open(output_file, 'wb') as f:
        # We need to write a magic number or simple versioning just in case
        f.write(np.uint32(0xCAFE0001).tobytes())
        
        # 1. Downsampling blocks
        write_convbnrelu(f, model.rcbn1, "rcbn1")
        write_convbnrelu(f, model.rcbn2, "rcbn2")
        write_convbnrelu(f, model.rcbn3, "rcbn3")
        
        # 2. Upsampling blocks
        write_up(f, model.up1, "up1")
        write_up(f, model.up2, "up2")
        
        # 3. Output sections
        write_conv1d(f, model.out_intermediate, "out_intermediate")
        write_conv1d(f, model.outc, "outc")

    print(f"Successfully wrote {output_file}.")

if __name__ == "__main__":
    main()
