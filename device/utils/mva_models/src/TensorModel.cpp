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
#include "TensorModel.h"
#include "BackendCommon.h"

#include <fstream>
#include <sstream>

namespace {
  constexpr auto format = "allen-tensors/1";

  std::string shape_string(const std::vector<int>& shape)
  {
    std::ostringstream s;
    s << "[";
    for (size_t i = 0; i < shape.size(); ++i)
      s << (i ? ", " : "") << shape[i];
    s << "]";
    return s.str();
  }
} // namespace

Allen::MVAModels::TensorModel::TensorModel(std::string name, std::function<std::string()> path, std::string kind) :
  MVAModelBase(std::move(name), std::move(path)), m_kind(std::move(kind))
{}

Allen::MVAModels::TensorModel::TensorModel(std::string name, std::string path, std::string kind) :
  MVAModelBase(std::move(name), std::move(path)), m_kind(std::move(kind))
{}

void Allen::MVAModels::TensorModel::readData(std::string parameters_path)
{
  m_file = file_path(parameters_path);
  std::ifstream in {m_file};
  if (!in) {
    throw StrException(
      m_name + ": cannot open the model file " + m_file +
      " (relative to the parameters directory --params, or absolute)");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(in);
  } catch (const nlohmann::json::exception& e) {
    throw StrException(m_name + ": " + m_file + " is not valid JSON: " + e.what());
  }
  if (!j.is_object() || j.value("format", "") != format) {
    throw StrException(m_name + ": " + m_file + " is not a tensor model file (format " + format + ")");
  }
  const std::string kind = j.value("kind", "");
  if (!m_kind.empty() && kind != m_kind) {
    throw StrException(m_name + ": " + m_file + " is a \"" + kind + "\" model, not \"" + m_kind + "\"");
  }
  m_kind = kind;
  m_metadata = j.value("metadata", nlohmann::json::object());

  try {
    for (const auto& [name, t] : j.at("tensors").items()) {
      Tensor tensor;
      tensor.shape = t.at("shape").get<std::vector<int>>();
      tensor.data = t.at("data").get<std::vector<float>>();
      size_t n = 1;
      for (const int d : tensor.shape)
        n *= static_cast<size_t>(d);
      if (n != tensor.data.size()) {
        throw StrException(
          m_name + ": " + m_file + ": tensor " + name + " has " + std::to_string(tensor.data.size()) +
          " values for shape " + shape_string(tensor.shape));
      }
      m_tensors.insert_or_assign(name, std::move(tensor));
    }
  } catch (const nlohmann::json::exception& e) {
    throw StrException(m_name + ": " + m_file + ": bad tensors: " + e.what());
  }
}

const Allen::MVAModels::TensorModel::Tensor& Allen::MVAModels::TensorModel::get(
  const std::string& name,
  const std::vector<int>& shape) const
{
  const auto it = m_tensors.find(name);
  if (it == m_tensors.end()) {
    throw StrException(m_name + ": " + m_file + " has no tensor " + name);
  }
  if (it->second.shape != shape) {
    throw StrException(
      m_name + ": " + m_file + ": tensor " + name + " has shape " + shape_string(it->second.shape) +
      ", this build expects " + shape_string(shape) + (m_shape_hint.empty() ? "" : " " + m_shape_hint));
  }
  return it->second;
}

Allen::MVAModels::TensorModel::Tensor& Allen::MVAModels::TensorModel::get(
  const std::string& name,
  const std::vector<int>& shape)
{
  return const_cast<Tensor&>(static_cast<const TensorModel*>(this)->get(name, shape));
}

const std::vector<float>& Allen::MVAModels::TensorModel::tensor(const std::string& name, const std::vector<int>& shape)
  const
{
  return get(name, shape).data;
}

const float* Allen::MVAModels::TensorModel::device_tensor(const std::string& name, const std::vector<int>& shape)
{
  Tensor& t = get(name, shape);
  if (t.device == nullptr) {
    Allen::malloc(reinterpret_cast<void**>(&t.device), t.data.size() * sizeof(float));
    Allen::memcpy(t.device, t.data.data(), t.data.size() * sizeof(float), Allen::memcpyHostToDevice);
  }
  return t.device;
}
