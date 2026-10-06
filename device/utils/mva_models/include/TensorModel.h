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
#include "nlohmann/json.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace Allen::MVAModels {

  // The trained parameters of a network as named float32 tensors, for
  // algorithms that run the network themselves (their own kernels, or
  // Allen::CuDNN layers). One JSON file, read by MVAModelsManager like the
  // other models:
  //
  //   {"format": "allen-tensors/1",
  //    "kind": "pvfinder",                      <- which network, checked
  //    "name": "...",                           <- free text, for logs
  //    "metadata": {"bn_eps": 1e-5, ...},       <- numbers or strings
  //    "tensors": {"layer1.weight": {"shape": [20, 9], "data": [...]}, ...}}
  //
  // Tensor names are free (for example a PyTorch state dict's), data are row
  // major (C order). Other top-level keys (provenance) are ignored.
  //
  // An algorithm declares it as a member, usually with the file from a
  // property, and takes the tensors in init(), which also checks their
  // shapes against what the algorithm was built for:
  //
  //   Allen::Property<std::string> m_model_file {this, "model", "my_net/model.json", "model file"};
  //   Allen::MVAModels::TensorModel m_model {"my_net", [this] { return m_model_file.value(); }, "my_net"};
  //   ...
  //   const float* w = m_model.device_tensor("conv1.weight", {16, 4, 25});   // in init()
  struct TensorModel : MVAModelBase {
    // path: the file, relative to the parameters directory or absolute (see
    // MVAModelBase). kind: the "kind" the file must have (empty: any).
    TensorModel(std::string name, std::function<std::string()> path, std::string kind);
    TensorModel(std::string name, std::string path, std::string kind);

    void readData(std::string parameters_path) override;

    bool has_tensor(const std::string& name) const { return m_tensors.count(name) != 0; }
    // A tensor on the host, after checking its shape.
    const std::vector<float>& tensor(const std::string& name, const std::vector<int>& shape) const;
    // A tensor on the device, after checking its shape: copied on the first
    // request and kept for the process lifetime. Not thread safe: call it from
    // init().
    const float* device_tensor(const std::string& name, const std::vector<int>& shape);

    // A metadata value; throws StrException when it is missing or not a T.
    template<typename T>
    T metadata(const std::string& key) const
    {
      const auto it = m_metadata.find(key);
      if (it == m_metadata.end()) {
        throw StrException(m_name + ": " + m_file + " has no metadata " + key);
      }
      try {
        return it->get<T>();
      } catch (const nlohmann::json::exception&) {
        throw StrException(m_name + ": " + m_file + ": metadata " + key + " has the wrong type (" + it->dump() + ")");
      }
    }

    // The file read, with the parameters directory resolved.
    const std::string& file() const { return m_file; }
    const std::string& kind() const { return m_kind; }

  protected:
    // Appended to shape mismatch errors: how to get a build that matches the
    // model, when the shapes are fixed at compile time.
    std::string m_shape_hint;

  private:
    struct Tensor {
      std::vector<int> shape;
      std::vector<float> data;
      float* device = nullptr;
    };
    Tensor& get(const std::string& name, const std::vector<int>& shape);
    const Tensor& get(const std::string& name, const std::vector<int>& shape) const;

    std::string m_kind;
    std::string m_file;
    nlohmann::json m_metadata = nlohmann::json::object();
    std::map<std::string, Tensor> m_tensors;
  };

} // namespace Allen::MVAModels
