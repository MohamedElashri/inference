/*****************************************************************************\
* (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

namespace MagneticField {
  struct Magfield {
#ifdef MAGFIELD_USE_TEXTURE
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory

#ifdef __CUDA_ARCH__
    __device__ float3 fieldVectorLinearInterpolation(float3 pos) const
    {
      const float x = (pos.x - minX) * invDx;
      const float y = (pos.y - minY) * invDy;
      const float z = (pos.z - minZ) * invDz;
      return {tex3D<float>(tex_Bx, x, y, z), tex3D<float>(tex_By, x, y, z), tex3D<float>(tex_Bz, x, y, z)};
    }
#else
    __device__ float3 fieldVectorLinearInterpolation(float3 pos) const;
#endif

    cudaArray_t array_Bx;
    cudaArray_t array_By;
    cudaArray_t array_Bz;

    cudaTextureObject_t tex_Bx;
    cudaTextureObject_t tex_By;
    cudaTextureObject_t tex_Bz;
#else
    __device__ float3 fieldVectorLinearInterpolation(float3 pos) const
    {
      const float x = (pos.x - minX) * invDx;
      const float y = (pos.y - minY) * invDy;
      const float z = (pos.z - minZ) * invDz;

      const int ix = (int) x;
      const int iy = (int) y;
      const int iz = (int) z;

      if (ix < 0 || iy < 0 || iz < 0 || ix >= Nx - 1 || iy >= Ny - 1 || iz >= Nz - 1) {
        return {0.f, 0.f, 0.f};
      }

      const int i000 = Nx * (Ny * iz + iy) + ix;
      const int i001 = Nx * (Ny * (iz + 1) + iy) + ix;
      const int i010 = Nx * (Ny * iz + iy + 1) + ix;
      const int i011 = Nx * (Ny * (iz + 1) + iy + 1) + ix;

      const float h1x = x - ix;
      const float h1y = y - iy;
      const float h1z = z - iz;

      const float h0x = 1.0f - h1x;
      const float h0y = 1.0f - h1y;
      const float h0z = 1.0f - h1z;

      const float h00 = h0x * h0y;
      const float h01 = h0x * h1y;
      const float h10 = h1x * h0y;
      const float h11 = h1x * h1y;

      return {(h0z * (h00 * Bx[i000] + h10 * Bx[i000 + 1] + h01 * Bx[i010] + h11 * Bx[i010 + 1]) +
               h1z * (h00 * Bx[i001] + h10 * Bx[i001 + 1] + h01 * Bx[i011] + h11 * Bx[i011 + 1])),

              (h0z * (h00 * By[i000] + h10 * By[i000 + 1] + h01 * By[i010] + h11 * By[i010 + 1]) +
               h1z * (h00 * By[i001] + h10 * By[i001 + 1] + h01 * By[i011] + h11 * By[i011 + 1])),

              (h0z * (h00 * Bz[i000] + h10 * Bz[i000 + 1] + h01 * Bz[i010] + h11 * Bz[i010 + 1]) +
               h1z * (h00 * Bz[i001] + h10 * Bz[i001 + 1] + h01 * Bz[i011] + h11 * Bz[i011 + 1]))};
    }

    float* Bx;
    float* By;
    float* Bz;

    int Nx, Ny, Nz;
#endif

    float invDx, invDy, invDz;
    float minX, minY, minZ;
  };

} // namespace MagneticField
