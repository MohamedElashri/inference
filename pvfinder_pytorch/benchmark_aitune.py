import os
import sys
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from utils import TrackIntervalsToKDE_HDplusUNet100 as Model
from utils import collect_t2kde_arrays
from utils import select_gpu

device = select_gpu(2) # GPU 2 is the RTX 3090
print(f"Using device: {device}")

# 1. Load data
print("Loading data...")
data_path = os.path.join(script_dir, "data/pv_HLT1CPU_MinBiasMagDown_14Nov_t2hists_Arrays_validation_allEvents.npy")
validation = collect_t2kde_arrays(
    data_path,
    batch_size=64,
    pin_memory=True,
    shuffle=False
)

vdt0 = validation.dataset.tensors[0]
# Split dataset into smaller chunks
nSplit = []
for ii in range(256 * 4): # Smaller batches (2000 events max)
    nSplit.append((ii+1)*2000)
vdt0Split = torch.tensor_split(vdt0, nSplit, dim=0)

# Use the first chunk as input, move to device
sample_input = vdt0Split[0].to(device)
print(f"Sample input shape: {sample_input.shape}")

import time

# 2. Load model
print("Loading model...")
nOut1, nOut2, nOut3, nOut4, nOut5 = 20, 20, 20, 20, 20
latentChannels = 8
nUNetChannels = 64
model = Model(nOut1, nOut2, nOut3, nOut4, nOut5, latentChannels=latentChannels, n=nUNetChannels)

weights_path = os.path.join(script_dir, 'weights/07Sept2023_t2hists_HDplusUNet100_iter12Ca_200epochs_2em5_5p0_final.pyt')
d = torch.load(weights_path, map_location='cpu')
model.load_state_dict(d)
model.to(device)
model.eval()

print("\n--- Baseline (Vanilla PyTorch) Evaluation ---")
# Warmup baseline
for _ in range(5):
    _ = model(sample_input)

if device.type == 'cuda':
    torch.cuda.synchronize()
start_base = time.perf_counter()

# Run the entire dataset through the baseline model
with torch.no_grad():
    for chunk in vdt0Split:
        chunk = chunk.to(device)
        _ = model(chunk)

if device.type == 'cuda':
    torch.cuda.synchronize()
end_base = time.perf_counter()
print(f"Vanilla PyTorch - Total time for all {vdt0.shape[0]} events: {end_base - start_base:.4f}s")


# 3. Setup AITune
print("\n--- AITune Evaluation ---")
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

import aitune.torch as ait
from aitune.torch.tune_strategy import HighestThroughputStrategy
from aitune.torch.backend import TensorRTBackend, TorchInductorBackend
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackendConfig

# We use HighestThroughputStrategy to find the fastest inference backend
strategy = HighestThroughputStrategy(
    backends=[TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)), TorchInductorBackend()]
)

model = ait.Module(model, strategy=strategy)

def inference_fn(x):
    return model(x)

print("Tuning model... This might take a while.")
ds = torch.utils.data.TensorDataset(sample_input)
ait.tune(
    func=inference_fn,
    dataset=ds,
    batch_sizes=[32]
)

print("Tuning completed successfully.")
# Perform a warmup and timed run to verify backend performance interactively
import time
for _ in range(5):
    _ = inference_fn(sample_input)

if device.type == 'cuda':
    torch.cuda.synchronize()
start_t = time.perf_counter()

# Run the entire dataset through the tuned model 
with torch.no_grad():
    for chunk in vdt0Split:
        chunk = chunk.to(device)
        _ = inference_fn(chunk)

if device.type == 'cuda':
    torch.cuda.synchronize()
end_t = time.perf_counter()

print(f"AITune (TorchInductorBackend) - Total time for all {vdt0.shape[0]} events: {end_t - start_t:.4f}s")
