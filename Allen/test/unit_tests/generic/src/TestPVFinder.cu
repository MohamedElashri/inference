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
// PVFinder (device/pvfinder): the peak finder on hand-made KDEs, its parallel
// version against the serial one, and the model file reader.
#if __has_include(<catch2/catch.hpp>)
// Catch2 v2
#include <catch2/catch.hpp>
namespace Catch {
  using Detail::Approx;
}
#else
// Catch2 v3
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "PVFinderModel.h"
#include "PVFinderPeakFinding.cuh"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace PVFinderConstants;
using PVFinderPeakFinding::Cuts;
using PVFinderPeakFinding::find_peaks_serial;

namespace {
  float bin_centre(float bin) { return KDE::z_min + KDE::bin_width * (bin + 0.5f); }

  struct Seeds {
    unsigned found = 0;
    std::vector<float> z;
  };

  Seeds serial(const std::vector<float>& kde, const Cuts& cuts = {})
  {
    std::vector<float> z(PV::max_number_vertices, 0.f);
    Seeds s;
    s.found = find_peaks_serial(kde.data(), z.data(), cuts);
    z.resize(std::min(s.found, PV::max_number_vertices));
    s.z = z;
    return s;
  }

  // A triangle of half-width w (bins) and height h centred on bin c: symmetric,
  // so its weighted mean is exactly c.
  void add_triangle(std::vector<float>& kde, unsigned c, unsigned w, float h)
  {
    for (unsigned i = c - w + 1; i < c + w; ++i) {
      const unsigned d = i > c ? i - c : c - i;
      kde[i] += h * static_cast<float>(w - d) / static_cast<float>(w);
    }
  }

  // Random events: peaks of random width and height at random bins (some
  // touching or overlapping, some at the ends), plus noise below and around
  // the threshold.
  std::vector<float> random_kdes(unsigned n_events, unsigned seed)
  {
    std::mt19937 g(seed);
    std::vector<float> kdes(size_t(n_events) * KDE::n_bins, 0.f);
    for (unsigned e = 0; e < n_events; ++e) {
      float* kde = kdes.data() + size_t(e) * KDE::n_bins;
      const unsigned n_peaks = std::uniform_int_distribution<unsigned>(0, e % 5 == 0 ? 45 : 12)(g);
      for (unsigned p = 0; p < n_peaks; ++p) {
        const float centre = std::uniform_real_distribution<float>(-5.f, KDE::n_bins + 5.f)(g);
        const float sigma = std::uniform_real_distribution<float>(0.5f, 8.f)(g);
        const float height = std::uniform_real_distribution<float>(0.02f, 1.5f)(g);
        for (int i = std::max(0, int(centre - 5 * sigma)); i < std::min(int(KDE::n_bins), int(centre + 5 * sigma));
             ++i) {
          const float d = (i + 0.5f - centre) / sigma;
          kde[i] += height * std::exp(-0.5f * d * d);
        }
      }
      std::uniform_real_distribution<float> noise(0.f, 0.08f);
      for (unsigned i = 0; i < KDE::n_bins; ++i) {
        if (g() % 16 == 0) kde[i] += noise(g);
      }
    }
    return kdes;
  }
} // namespace

TEST_CASE("pvfinder peak finding: hand-made KDEs", "[PVFinder]")
{
  std::vector<float> kde(KDE::n_bins, 0.f);

  SECTION("no peak")
  {
    REQUIRE(serial(kde).found == 0);
    kde[100] = 0.069f; // below threshold
    REQUIRE(serial(kde).found == 0);
  }

  SECTION("one peak: the weighted mean of its bin centres")
  {
    add_triangle(kde, 1234, 6, 0.5f);
    const auto s = serial(kde);
    REQUIRE(s.found == 1);
    REQUIRE(s.z[0] == Catch::Approx(bin_centre(1234)).margin(1e-4));
    // Asymmetric: bins 10, 11 with values 0.4, 0.6 -> mean bin 10.6.
    std::vector<float> two(KDE::n_bins, 0.f);
    two[10] = 0.4f;
    two[11] = 0.6f;
    REQUIRE(serial(two).z[0] == Catch::Approx(bin_centre(10.6f)).margin(1e-5));
  }

  SECTION("integral and width cuts")
  {
    kde[500] = 0.69f; // one bin, integral below 0.7
    REQUIRE(serial(kde).found == 0);
    kde[500] = 0.7f;
    REQUIRE(serial(kde).found == 1);
    Cuts wide;
    wide.min_width = 2;
    REQUIRE(serial(kde, wide).found == 0);
    Cuts loose;
    loose.integral_threshold = 0.1f;
    kde[800] = 0.1f;
    REQUIRE(serial(kde, loose).found == 2);
  }

  SECTION("separate peaks come out in increasing z")
  {
    add_triangle(kde, 3000, 5, 0.6f);
    add_triangle(kde, 200, 5, 0.6f);
    add_triangle(kde, 1500, 5, 0.6f);
    const auto s = serial(kde);
    REQUIRE(s.found == 3);
    REQUIRE(s.z[0] == Catch::Approx(bin_centre(200)).margin(1e-4));
    REQUIRE(s.z[1] == Catch::Approx(bin_centre(1500)).margin(1e-4));
    REQUIRE(s.z[2] == Catch::Approx(bin_centre(3000)).margin(1e-4));
  }

  SECTION("a run is split where the KDE rises again after a significant drop")
  {
    // All bins above threshold; 0.8 -> 0.1 is a drop of more than 0.05 and a
    // factor 1.1, and the KDE rises again at bin 23.
    const float values[] = {0.8f, 0.8f, 0.1f, 0.8f, 0.8f};
    for (unsigned i = 0; i < 5; ++i)
      kde[20 + i] = values[i];
    auto s = serial(kde);
    REQUIRE(s.found == 2);
    // As in pv-finder, the rising bin 23 still belongs to the first peak
    // (bins 20..23); the second is bin 24.
    REQUIRE(s.z[0] == Catch::Approx(bin_centre((20 * 0.8f + 21 * 0.8f + 22 * 0.1f + 23 * 0.8f) / 2.5f)).margin(1e-4));
    REQUIRE(s.z[1] == Catch::Approx(bin_centre(24)).margin(1e-4));
    Cuts no_split;
    no_split.split_peaks = false;
    REQUIRE(serial(kde, no_split).found == 1);
    // A small dip (0.8 -> 0.78) does not split.
    kde[22] = 0.78f;
    REQUIRE(serial(kde).found == 1);
  }

  SECTION("peaks at both ends of the range")
  {
    kde[0] = kde[1] = 0.5f;
    kde[KDE::n_bins - 2] = kde[KDE::n_bins - 1] = 0.5f;
    const auto s = serial(kde);
    REQUIRE(s.found == 2);
    REQUIRE(s.z[0] == Catch::Approx(bin_centre(0.5f)).margin(1e-4));
    REQUIRE(s.z[1] == Catch::Approx(bin_centre(KDE::n_bins - 1.5f)).margin(1e-4));
  }

  SECTION("at most PV::max_number_vertices seeds, the lowest in z")
  {
    const unsigned n = PV::max_number_vertices + 8;
    for (unsigned p = 0; p < n; ++p)
      add_triangle(kde, 50 + p * 90, 4, 0.5f);
    const auto s = serial(kde);
    REQUIRE(s.found == n);
    REQUIRE(s.z.size() == PV::max_number_vertices);
    REQUIRE(s.z.back() == Catch::Approx(bin_centre(50 + (PV::max_number_vertices - 1) * 90)).margin(1e-4));
  }
}

#ifdef TARGET_DEVICE_CUDA
#include <cuda_runtime.h>

namespace {
  __global__ void find_peaks_kernel(const float* kdes, float* zpeaks, unsigned* found, const Cuts cuts)
  {
    const unsigned n = PVFinderPeakFinding::find_peaks(
      kdes + size_t(blockIdx.x) * KDE::n_bins, zpeaks + blockIdx.x * PV::max_number_vertices, cuts);
    if (threadIdx.x == 0) found[blockIdx.x] = n;
  }
} // namespace

TEST_CASE("pvfinder peak finding: parallel = serial", "[PVFinder]")
{
  int n_devices = 0;
  if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
    WARN("no CUDA device: skipped");
    return;
  }
  const unsigned n_events = 400;
  const auto kdes = random_kdes(n_events, 7);
  float* d_kdes = nullptr;
  float* d_zpeaks = nullptr;
  unsigned* d_found = nullptr;
  REQUIRE(cudaMalloc(&d_kdes, kdes.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_zpeaks, n_events * PV::max_number_vertices * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_found, n_events * sizeof(unsigned)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_kdes, kdes.data(), kdes.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

  for (const bool split : {true, false}) {
    Cuts cuts;
    cuts.split_peaks = split;
    // Block sizes that split the 4000 bins evenly and unevenly, down to one
    // bin per thread, so runs cross the threads' shares.
    for (const unsigned block : {1u, 32u, 96u, 128u, 500u, 512u}) {
      REQUIRE(cudaMemset(d_zpeaks, 0, n_events * PV::max_number_vertices * sizeof(float)) == cudaSuccess);
      find_peaks_kernel<<<n_events, block>>>(d_kdes, d_zpeaks, d_found, cuts);
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      std::vector<float> zpeaks(n_events * PV::max_number_vertices);
      std::vector<unsigned> found(n_events);
      REQUIRE(
        cudaMemcpy(zpeaks.data(), d_zpeaks, zpeaks.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(
        cudaMemcpy(found.data(), d_found, found.size() * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);

      unsigned mismatched_counts = 0, truncated = 0, total = 0;
      float max_dz = 0.f;
      for (unsigned e = 0; e < n_events; ++e) {
        const std::vector<float> kde(
          kdes.begin() + size_t(e) * KDE::n_bins, kdes.begin() + size_t(e + 1) * KDE::n_bins);
        const auto ref = serial(kde, cuts);
        total += ref.found;
        truncated += ref.found > PV::max_number_vertices;
        if (found[e] != ref.found) {
          ++mismatched_counts;
          continue;
        }
        for (size_t i = 0; i < ref.z.size(); ++i)
          max_dz = std::max(max_dz, std::fabs(zpeaks[e * PV::max_number_vertices + i] - ref.z[i]));
      }
      INFO("block " << block << ", split " << split << ": " << total << " peaks, " << truncated << " truncated events");
      REQUIRE(total > 1000);
      REQUIRE(truncated > 0);
      REQUIRE(mismatched_counts == 0);
      // Same sums in the same order; only FMA contraction may differ.
      REQUIRE(max_dz < 1e-4f);
    }
  }
  cudaFree(d_kdes);
  cudaFree(d_zpeaks);
  cudaFree(d_found);
}
#endif

namespace {
  struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory()
    {
      path =
        std::filesystem::temp_directory_path() / ("allen_test_pvfinder_" + std::to_string(std::random_device {}()));
      std::filesystem::create_directories(path);
    }
    ~TemporaryDirectory() { std::filesystem::remove_all(path); }
    std::string write(const std::string& name, const std::string& content) const
    {
      std::ofstream(path / name) << content;
      return (path / name).string();
    }
  };

  const std::string valid_model = R"({"format": "pvfinder-model/1", "name": "test", "latent_channels": 4,
    "unet_features": 16, "bn_eps": 1e-5, "tensors": {
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

TEST_CASE("pvfinder model file", "[PVFinder]")
{
  const TemporaryDirectory dir;
  std::string path;
  PVFinder::Model model {"test_model", [&] { return path; }};

  SECTION("a valid file, absolute path")
  {
    path = dir.write("model.json", valid_model);
    model.readData("/nonexistent");
    REQUIRE(model.file() == path);
    REQUIRE(model.latent_channels() == 4);
    REQUIRE(model.unet_features() == 16);
    REQUIRE(model.bn_eps() == 1e-5f);
    REQUIRE(model.tensor("layer1.weight", {2, 3}) == std::vector<float> {1, 2, 3, 4, 5, 6.5f});
    REQUIRE(model.tensor("layer1.bias", {2}) == std::vector<float> {-1, 0.25f});
  }

  SECTION("a relative path is in the parameters directory")
  {
    std::filesystem::create_directories(dir.path / "pvfinder");
    dir.write("pvfinder/model.json", valid_model);
    path = "pvfinder/model.json";
    for (const std::string params : {dir.path.string(), dir.path.string() + "/"}) {
      PVFinder::Model m {"test_model", [&] { return path; }};
      m.readData(params);
      REQUIRE(m.file() == (dir.path / "pvfinder/model.json").string());
      REQUIRE(m.tensor("layer1.bias", {2}).size() == 2);
    }
  }

  SECTION("wrong requests")
  {
    path = dir.write("model.json", valid_model);
    model.readData("");
    REQUIRE(contains(error_of([&] { model.tensor("layer1.weight", {3, 2}); }), "this build expects [3, 2]"));
    REQUIRE(contains(error_of([&] { model.tensor("layer2.weight", {2, 3}); }), "has no tensor layer2.weight"));
  }

  SECTION("bad files")
  {
    path = (dir.path / "missing.json").string();
    REQUIRE(contains(error_of([&] { model.readData(""); }), "cannot open"));
    path = dir.write("broken.json", "{\"format\": ");
    REQUIRE(contains(error_of([&] { model.readData(""); }), "not valid JSON"));
    path = dir.write("other.json", R"({"format": "something-else/1"})");
    REQUIRE(contains(error_of([&] { model.readData(""); }), "is not a PVFinder model"));
    path = dir.write(
      "short.json",
      R"({"format": "pvfinder-model/1", "latent_channels": 4, "unet_features": 16, "bn_eps": 1e-5,
          "tensors": {"w": {"shape": [2, 2], "data": [1, 2, 3]}}})");
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
