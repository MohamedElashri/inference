/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <string>

#include <BackendCommon.h>
#include <Common.h>
#include <Consumers.h>
#include "MagneticField.cuh"

namespace {
  using std::string;
  using std::to_string;
} // namespace

Consumers::MagneticFieldPolarity::MagneticFieldPolarity(
  std::span<float>& dev_magnet_polarity,
  std::vector<float>& host_magnet_polarity) :
  m_dev_magnet_polarity {dev_magnet_polarity},
  m_host_magnet_polarity {host_magnet_polarity}
{}

void Consumers::MagneticFieldPolarity::consume(std::vector<char> const& data)
{
  auto& dev_magnet_polarity = m_dev_magnet_polarity.get();
  auto& host_magnet_polarity = m_host_magnet_polarity.get();
  if (dev_magnet_polarity.empty()) {
    // Allocate space
    float* p = nullptr;
    Allen::malloc((void**) &p, data.size());
    dev_magnet_polarity = {p, static_cast<span_size_t<char>>(data.size() / sizeof(float))};
    host_magnet_polarity.resize(data.size() / sizeof(float));
  }
  else if (data.size() != static_cast<size_t>(sizeof(float) * dev_magnet_polarity.size())) {
    throw StrException {string {"sizes don't match: "} + to_string(dev_magnet_polarity.size()) + " " +
                        to_string(data.size() / sizeof(float))};
  }

  Allen::memcpy(dev_magnet_polarity.data(), data.data(), data.size(), Allen::memcpyHostToDevice);
  Allen::memcpy(host_magnet_polarity.data(), data.data(), data.size(), Allen::memcpyHostToHost);
}

Consumers::MagneticField::MagneticField(Constants& constants) : m_constants {constants} {}

void Consumers::MagneticField::consume(std::vector<char> const& data)
{
  // Parse header
  const float* invDxyz = reinterpret_cast<const float*>(data.data());
  const int* Nxyz = reinterpret_cast<const int*>(data.data() + sizeof(float) * 4);
  const float* min = reinterpret_cast<const float*>(data.data() + sizeof(float) * 8);
  const float* B = reinterpret_cast<const float*>(data.data() + sizeof(float) * 12);

  printf("Loading magfield grid, nbins x,y,z : (%d, %d, %d)\n", Nxyz[0], Nxyz[1], Nxyz[2]);
  printf(
    "dx, xmin, xmax: (%f, %f, %f)\n",
    (double) (1.f / invDxyz[0]),
    (double) min[0],
    (double) (min[0] + (Nxyz[0] - 1) / invDxyz[0]));
  printf(
    "dy, ymin, ymax: (%f, %f, %f)\n",
    (double) (1.f / invDxyz[1]),
    (double) min[1],
    (double) (min[1] + (Nxyz[1] - 1) / invDxyz[1]));
  printf(
    "dz, zmin, zmax: (%f, %f, %f)\n",
    (double) (1.f / invDxyz[2]),
    (double) min[2],
    (double) (min[2] + (Nxyz[2] - 1) / invDxyz[2]));

  // Reorder B field
  std::size_t N = Nxyz[0] * Nxyz[1] * Nxyz[2];
  std::vector<float> Bx, By, Bz;
  Bx.reserve(N);
  By.reserve(N);
  Bz.reserve(N);
  for (std::size_t i = 0; i < N * 4; i += 4) {
    Bx.emplace_back(B[i]);
    By.emplace_back(B[i + 1]);
    Bz.emplace_back(B[i + 2]);
  }

  auto& magnetic_field = m_constants.get().magnetic_field;
  if (magnetic_field == nullptr) {
    magnetic_field = new ::MagneticField::Magfield();
    magnetic_field->invDx = invDxyz[0];
    magnetic_field->invDy = invDxyz[1];
    magnetic_field->invDz = invDxyz[2];
    magnetic_field->minX = min[0];
    magnetic_field->minY = min[1];
    magnetic_field->minZ = min[2];

#ifdef MAGFIELD_USE_TEXTURE
    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc chDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExtent extent;
    extent.width = Nxyz[0];
    extent.height = Nxyz[1];
    extent.depth = Nxyz[2];
    cudaCheck(cudaMalloc3DArray(&magnetic_field->array_Bx, &chDesc, extent));
    cudaCheck(cudaMalloc3DArray(&magnetic_field->array_By, &chDesc, extent));
    cudaCheck(cudaMalloc3DArray(&magnetic_field->array_Bz, &chDesc, extent));

    // Specify texture object parameters
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Specify textures
    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    resDesc.res.array.array = magnetic_field->array_Bx;
    magnetic_field->tex_Bx = 0;
    cudaCheck(cudaCreateTextureObject(&magnetic_field->tex_Bx, &resDesc, &texDesc, NULL));

    resDesc.res.array.array = magnetic_field->array_By;
    magnetic_field->tex_By = 0;
    cudaCheck(cudaCreateTextureObject(&magnetic_field->tex_By, &resDesc, &texDesc, NULL));

    resDesc.res.array.array = magnetic_field->array_Bz;
    magnetic_field->tex_Bz = 0;
    cudaCheck(cudaCreateTextureObject(&magnetic_field->tex_Bz, &resDesc, &texDesc, NULL));
#else
    magnetic_field->Nx = Nxyz[0];
    magnetic_field->Ny = Nxyz[1];
    magnetic_field->Nz = Nxyz[2];

    Allen::malloc((void**) &magnetic_field->Bx, sizeof(float) * N);
    Allen::malloc((void**) &magnetic_field->By, sizeof(float) * N);
    Allen::malloc((void**) &magnetic_field->Bz, sizeof(float) * N);
#endif
  }

  // FIXME need to check the size of data is as expected

#ifdef MAGFIELD_USE_TEXTURE

  cudaMemcpy3DParms cpyParms;
  memset(&cpyParms, 0, sizeof(cpyParms));
  cpyParms.srcPtr.pitch = Nxyz[0] * sizeof(float);
  cpyParms.srcPtr.xsize = Nxyz[0];
  cpyParms.srcPtr.ysize = Nxyz[1];
  cpyParms.kind = cudaMemcpyHostToDevice;
  cpyParms.extent.width = Nxyz[0];
  cpyParms.extent.height = Nxyz[1];
  cpyParms.extent.depth = Nxyz[2];

  cpyParms.srcPtr.ptr = Bx.data();
  cpyParms.dstArray = magnetic_field->array_Bx;
  cudaCheck(cudaMemcpy3D(&cpyParms));

  cpyParms.srcPtr.ptr = By.data();
  cpyParms.dstArray = magnetic_field->array_By;
  cudaCheck(cudaMemcpy3D(&cpyParms));

  cpyParms.srcPtr.ptr = Bz.data();
  cpyParms.dstArray = magnetic_field->array_Bz;
  cudaCheck(cudaMemcpy3D(&cpyParms));

#else
  Allen::memcpy(magnetic_field->Bx, Bx.data(), sizeof(float) * N, Allen::memcpyHostToDevice);
  Allen::memcpy(magnetic_field->By, By.data(), sizeof(float) * N, Allen::memcpyHostToDevice);
  Allen::memcpy(magnetic_field->Bz, Bz.data(), sizeof(float) * N, Allen::memcpyHostToDevice);
#endif
}
