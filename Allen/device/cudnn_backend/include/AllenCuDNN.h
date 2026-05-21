// (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
// Apache-2.0 License
//
// AllenCuDNN — translational layer between Allen and cuDNN.
// Include this header in any DeviceAlgorithm that uses cuDNN.
//
// Link against the AllenCuDNN CMake target.
#pragma once

#include "CuDNNBackendShim.h"   // compile-time backend selection
#include "CuDNNCheck.h"         // ALLEN_CUDNN_CHECK macro
#include "CuDNNHandle.h"        // RAII logical stream routing handle
#include "CuDNNWorkspace.h"     // Explicit external workspace handle
#include "CuDNNDescriptors.h"   // Convolution shape configurations caching
#include "CuDNNFusedOps.h"      // Metadata-first fused convolution API
#include "CuDNNDeviceWeights.h" // Process-lifetime typed device weights
#include "CuDNNWeightRegistry.h"// Legacy singleton facade over DeviceWeights
#include "CuDNNLayoutTransform.cuh"// Boilerplate macros
