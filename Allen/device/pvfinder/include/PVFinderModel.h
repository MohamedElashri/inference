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

#include "MVAModelsManager.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace PVFinder {

  // A trained PVFinder model (FC network and UNet): one JSON file, read by
  // Allen's MVAModelsManager like the other Allen models,
  //   {"format": "pvfinder-model/1", "name": ..., "latent_channels": 4,
  //    "unet_features": 16, "tensors": {"layer1.weight": {"shape": [20, 9],
  //    "data": [...]}, ...}}
  // with the tensors named as in the PyTorch state dict, their data row major
  // (C order), and every float exactly the checkpoint's float32. The
  // algorithm's "model" property names the file: relative to the parameters
  // directory (--params), or absolute.
  struct Model : Allen::MVAModels::MVAModelBase {
    Model(std::string name, std::function<std::string()> path) : MVAModelBase(std::move(name), std::move(path)) {}

    void readData(std::string parameters_path) override;

    // A tensor on the host, after checking its shape.
    const std::vector<float>& tensor(const std::string& name, const std::vector<int>& shape) const;

    // A tensor on the device (copied on the first request, kept for the
    // process lifetime), after checking its shape. Not thread safe: call it
    // from init().
    const float* device_tensor(const std::string& name, const std::vector<int>& shape);

    const std::string& file() const { return m_file; }
    unsigned latent_channels() const { return m_latent_channels; }
    unsigned unet_features() const { return m_unet_features; }
    float bn_eps() const { return m_bn_eps; }

  private:
    struct Tensor {
      std::vector<int> shape;
      std::vector<float> data;
      float* device = nullptr;
    };
    Tensor& get(const std::string& name, const std::vector<int>& shape);
    const Tensor& get(const std::string& name, const std::vector<int>& shape) const;

    std::string m_file;
    unsigned m_latent_channels = 0;
    unsigned m_unet_features = 0;
    float m_bn_eps = 0.f;
    std::map<std::string, Tensor> m_tensors;
  };

} // namespace PVFinder
