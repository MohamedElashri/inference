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

#include "TensorModel.h"

namespace PVFinder {

  // A trained PVFinder model (FC network and UNet): a tensor model file
  // (Allen::MVAModels::TensorModel) of kind "pvfinder", with the tensors named
  // as in the PyTorch state dict and the metadata
  //   latent_channels  UNet input channels (FC outputs per bin)
  //   unet_features    UNet feature channels
  //   bn_eps           BatchNorm epsilon
  // The algorithm's "model" property names the file: relative to the
  // parameters directory (--params), or absolute.
  struct Model : Allen::MVAModels::TensorModel {
    Model(std::string name, std::function<std::string()> path) :
      TensorModel(std::move(name), std::move(path), "pvfinder")
    {
      m_shape_hint = "(the build's --unet-feat and --unet-batch-channels must match the model)";
    }

    void readData(std::string parameters_path) override
    {
      TensorModel::readData(parameters_path);
      m_latent_channels = metadata<unsigned>("latent_channels");
      m_unet_features = metadata<unsigned>("unet_features");
      m_bn_eps = metadata<float>("bn_eps");
    }

    unsigned latent_channels() const { return m_latent_channels; }
    unsigned unet_features() const { return m_unet_features; }
    float bn_eps() const { return m_bn_eps; }

  private:
    unsigned m_latent_channels = 0;
    unsigned m_unet_features = 0;
    float m_bn_eps = 0.f;
  };

} // namespace PVFinder
