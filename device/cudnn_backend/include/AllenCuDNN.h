/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

// Allen's cuDNN library (built with WITH_CUDNN=ON, CUDA only; defines
// ALLEN_WITH_CUDNN): per-stream handles, graphs and plans on the cuDNN graph
// API, and CNN layers built from them. See doc/develop/allen_cudnn.rst.
#include "CuDNNCheck.h"
#include "CuDNNHandle.h"
#include "CuDNNGraph.h"
#include "CuDNNLayers.h"
