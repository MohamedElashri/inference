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
// Allen::MVAModels::TensorModel (device/utils/mva_models): tensor model files.
#if __has_include(<catch2/catch.hpp>)
#include <catch2/catch.hpp>
#else
#include <catch2/catch_test_macros.hpp>
#endif

#include "TensorModel.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifdef TARGET_DEVICE_CUDA
#include <cuda_runtime.h>
#endif

using Allen::MVAModels::TensorModel;

namespace {
  struct TemporaryDirectory {
    std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("allen_test_tensor_model_" + std::to_string(std::random_device {}()));
    TemporaryDirectory() { std::filesystem::create_directories(path); }
    ~TemporaryDirectory() { std::filesystem::remove_all(path); }
    std::string write(const std::string& name, const std::string& content) const
    {
      std::ofstream(path / name) << content;
      return (path / name).string();
    }
  };

  const std::string valid_model = R"({"format": "allen-tensors/1", "kind": "test_net", "name": "test",
    "source": "ignored", "metadata": {"channels": 4, "eps": 1e-5, "activation": "relu"}, "tensors": {
      "layer1.weight": {"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6.5]},
      "layer1.bias": {"shape": [2], "data": [-1, 0.25]}}})";

  // The message of the StrException that f throws ("" if none).
  template<typename F>
  std::string error_of(F&& f)
  {
    try {
      f();
    } catch (const StrException& e) {
      return e.what();
    }
    return "";
  }

  bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
} // namespace

TEST_CASE("tensor model file", "[TensorModel]")
{
  const TemporaryDirectory dir;
  std::string path;
  TensorModel model {"test_model", [&] { return path; }, "test_net"};

  SECTION("a valid file, absolute path")
  {
    path = dir.write("model.json", valid_model);
    model.readData("/nonexistent");
    REQUIRE(model.file() == path);
    REQUIRE(model.kind() == "test_net");
    REQUIRE(model.has_tensor("layer1.bias"));
    REQUIRE(!model.has_tensor("layer2.bias"));
    REQUIRE(model.tensor("layer1.weight", {2, 3}) == std::vector<float> {1, 2, 3, 4, 5, 6.5f});
    REQUIRE(model.tensor("layer1.bias", {2}) == std::vector<float> {-1, 0.25f});
    REQUIRE(model.metadata<unsigned>("channels") == 4);
    REQUIRE(model.metadata<float>("eps") == 1e-5f);
    REQUIRE(model.metadata<std::string>("activation") == "relu");
  }

  SECTION("a relative path is in the parameters directory")
  {
    std::filesystem::create_directories(dir.path / "net");
    dir.write("net/model.json", valid_model);
    path = "net/model.json";
    for (const std::string params : {dir.path.string(), dir.path.string() + "/"}) {
      TensorModel m {"test_model", [&] { return path; }, "test_net"};
      m.readData(params);
      REQUIRE(m.file() == (dir.path / "net/model.json").string());
      REQUIRE(m.tensor("layer1.bias", {2}).size() == 2);
    }
  }

  SECTION("any kind when none is required")
  {
    path = dir.write("model.json", valid_model);
    TensorModel any {"test_model", [&] { return path; }, ""};
    any.readData("");
    REQUIRE(any.kind() == "test_net");
  }

  SECTION("wrong requests")
  {
    path = dir.write("model.json", valid_model);
    model.readData("");
    REQUIRE(contains(error_of([&] { model.tensor("layer1.weight", {3, 2}); }), "this build expects [3, 2]"));
    REQUIRE(contains(error_of([&] { model.tensor("layer2.weight", {2, 3}); }), "has no tensor layer2.weight"));
    REQUIRE(contains(error_of([&] { model.metadata<float>("missing"); }), "has no metadata missing"));
    REQUIRE(contains(error_of([&] { model.metadata<float>("activation"); }), "metadata activation has the wrong type"));
  }

  SECTION("bad files")
  {
    path = (dir.path / "missing.json").string();
    REQUIRE(contains(error_of([&] { model.readData(""); }), "cannot open"));
    path = dir.write("broken.json", "{\"format\": ");
    REQUIRE(contains(error_of([&] { model.readData(""); }), "not valid JSON"));
    path = dir.write("other.json", R"({"format": "something-else/1"})");
    REQUIRE(contains(error_of([&] { model.readData(""); }), "is not a tensor model file"));
    path = dir.write("kind.json", R"({"format": "allen-tensors/1", "kind": "other_net", "tensors": {}})");
    REQUIRE(contains(error_of([&] { model.readData(""); }), "is a \"other_net\" model, not \"test_net\""));
    path = dir.write("notensors.json", R"({"format": "allen-tensors/1", "kind": "test_net"})");
    REQUIRE(contains(error_of([&] { model.readData(""); }), "bad tensors"));
    path = dir.write(
      "short.json",
      R"({"format": "allen-tensors/1", "kind": "test_net", "tensors": {"w": {"shape": [2, 2], "data": [1, 2, 3]}}})");
    REQUIRE(contains(error_of([&] { model.readData(""); }), "has 3 values for shape [2, 2]"));
  }

#ifdef TARGET_DEVICE_CUDA
  SECTION("device tensors")
  {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
      WARN("no CUDA device: skipped");
      return;
    }
    path = dir.write("model.json", valid_model);
    model.readData("");
    const float* d = model.device_tensor("layer1.weight", {2, 3});
    REQUIRE(model.device_tensor("layer1.weight", {2, 3}) == d); // copied once
    std::vector<float> back(6);
    REQUIRE(cudaMemcpy(back.data(), d, 6 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(back == model.tensor("layer1.weight", {2, 3}));
  }
#endif
}
