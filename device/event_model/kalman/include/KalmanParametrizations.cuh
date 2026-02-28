/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "Common.h"
#include "BackendCommon.h"
#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"

#include <fstream>
#include <sstream>

namespace ParKalmanFilter {
  //----------------------------------------------------------------------
  // Structure for storing parameters and performing extrapolations.
  struct KalmanParametrizations {

    // Parameters.
    // StandardCoefs C[nBinXMax][nBinYMax];
    float x00[nBinXMax * nBinYMax * DEGx2];
    float x10[nBinXMax * nBinYMax * DEGx1];
    float x01[nBinXMax * nBinYMax * DEGx1];
    float tx00[nBinXMax * nBinYMax * DEGx2];
    float tx10[nBinXMax * nBinYMax * DEGx1];
    float tx01[nBinXMax * nBinYMax * DEGx1];

    float y00[nBinXMax * nBinYMax * DEGy2];
    float y10[nBinXMax * nBinYMax * DEGy1];
    float y01[nBinXMax * nBinYMax * DEGy1];
    float ty00[nBinXMax * nBinYMax * DEGy2];
    float ty10[nBinXMax * nBinYMax * DEGy1];
    float ty01[nBinXMax * nBinYMax * DEGy1];

    float ZINI, ZFIN;
    float PMIN;
    float BENDX, BENDX_X2, BENDX_Y2, BENDY_XY;
    float Txmax, Tymax, XFmax, Xmax, Ymax;
    float Dtxy;
    float step;

// Parameters for Pierre's UT-T extrapolation.
#define QUADRATICINTERPOLATION 1
    int Nbinx, Nbiny;
    int XGridOption, YGridOption;
    int DEGX1, DEGX2, DEGY1, DEGY2;

    // Keep track of polarity and whether or not parameters have been loaded.
    bool paramsLoaded = false;

    __device__ __host__ inline KalmanFloat UTTExtrEndZ() const;

    __device__ __host__ inline KalmanFloat UTTExtrBeginZ() const;

    // Pierre's extrapolation.
    __host__ inline void read_params_UTT(std::string file);

    //----------------------------------------------------------------------
    // Set parameters.
    __host__ void SetParameters(std::string param_file_location)
    {
      // Get polarity.
      if (paramsLoaded) return;
      std::string ParamFileLocation;
      ParamFileLocation = param_file_location + "/ParametrizedKalmanFit/25v0/";
      read_params_UTT(ParamFileLocation + "/params_UTT_v0.tab");
      paramsLoaded = true;
    }

    // Template helper to compute polynomial UTT extrapolation
    // DEG0: Polynomial degree for arr00 coefficients
    // DEG1: Polynomial degree for arr10/arr01 coefficients
    template<int DEG0, int DEG1>
    __device__ __host__ void compute_state(
      const float* __restrict__ arr00,
      const float* __restrict__ arr10,
      const float* __restrict__ arr01,
      const int ix,
      const int iy,
      const KalmanFloat gx,
      const KalmanFloat gy,
      const int sx,
      const int sy,
      const int rx,
      const int ry,
      const KalmanFloat ux,
      const KalmanFloat uy,
      const KalmanFloat fq,
      KalmanFloat& state_out,
      KalmanFloat& der_qop_out,
      KalmanFloat& der_tx_out,
      KalmanFloat& der_ty_out) const
    {
      // Precompute interpolation scalers (same for all 3 coefficient sets)
      const KalmanFloat scaler0 = ((gy * gy - gy) * ((KalmanFloat) 0.5) + (!ry) * sx * gx * gy);
      const KalmanFloat scaler1 = (((KalmanFloat) 1) - (gx * gx + gy * gy) + sx * sy * gx * gy);
      const KalmanFloat scaler2 = ((gy * gy + gy) * ((KalmanFloat) 0.5) - ry * sx * gx * gy);
      const KalmanFloat scaler3 = ((gx * gx - gx) * ((KalmanFloat) 0.5) + (!rx) * sy * gx * gy);
      const KalmanFloat scaler4 = (sx * sy * gx * gy);
      const KalmanFloat scaler5 = ((gx * gx + gx) * ((KalmanFloat) 0.5) - rx * sy * gx * gy);

      // Process 00 coefficients (degree DEG0)
      KalmanFloat ff = 1;
      for (int deg = 0; deg < DEG0; deg++) {
        // Interpolate coefficient from 6 grid points
        const int bi_center = UttOffset(ix, iy, DEG0);
        const KalmanFloat coef =
          arr00[UttOffset(ix, iy - 1, DEG0) + deg] * scaler0 + arr00[bi_center + deg] * scaler1 +
          arr00[UttOffset(ix, iy + 1, DEG0) + deg] * scaler2 + arr00[UttOffset(ix - 1, iy, DEG0) + deg] * scaler3 +
          arr00[UttOffset(ix + sx, iy + sy, DEG0) + deg] * scaler4 + arr00[UttOffset(ix + 1, iy, DEG0) + deg] * scaler5;

        der_qop_out += (deg + 1) * coef * ff;
        ff *= fq;
        state_out += coef * ff;
      }

      // Process 10 coefficients (degree DEG1, ux term)
      ff = 1;
      for (int deg = 0; deg < DEG1; deg++) {
        const int bi_center = UttOffset(ix, iy, DEG1);
        const KalmanFloat coef =
          arr10[UttOffset(ix, iy - 1, DEG1) + deg] * scaler0 + arr10[bi_center + deg] * scaler1 +
          arr10[UttOffset(ix, iy + 1, DEG1) + deg] * scaler2 + arr10[UttOffset(ix - 1, iy, DEG1) + deg] * scaler3 +
          arr10[UttOffset(ix + sx, iy + sy, DEG1) + deg] * scaler4 + arr10[UttOffset(ix + 1, iy, DEG1) + deg] * scaler5;

        const KalmanFloat term = coef * ux;
        der_qop_out += (deg + 1) * term * ff;
        ff *= fq;
        state_out += term * ff;
        der_tx_out += coef * ff;
      }

      // Process 01 coefficients (degree DEG1, uy term)
      ff = 1;
      for (int deg = 0; deg < DEG1; deg++) {
        const int bi_center = UttOffset(ix, iy, DEG1);
        const KalmanFloat coef =
          arr01[UttOffset(ix, iy - 1, DEG1) + deg] * scaler0 + arr01[bi_center + deg] * scaler1 +
          arr01[UttOffset(ix, iy + 1, DEG1) + deg] * scaler2 + arr01[UttOffset(ix - 1, iy, DEG1) + deg] * scaler3 +
          arr01[UttOffset(ix + sx, iy + sy, DEG1) + deg] * scaler4 + arr01[UttOffset(ix + 1, iy, DEG1) + deg] * scaler5;

        const KalmanFloat term = coef * uy;
        der_qop_out += (deg + 1) * term * ff;
        ff *= fq;
        state_out += term * ff;
        der_ty_out += coef * ff;
      }
    }

    __host__ __device__ inline int UttOffset(const int& x_bin, const int& y_bin, const int& deg) const
    { // simple row-major access pattern
      return deg * (nBinYMax * x_bin + y_bin);
    }

    __host__ int Read(int ix, int iy, std::istream& inFile)
    {
      auto GetPar = [](std::istream& ss) -> float {
        std::string s;
        ss >> s;
        return std::stof(s);
      };
      int bi = UttOffset(ix, iy, DEGx2);
      for (int i = 0; i < DEGx2; i++) {
        this->x00[bi + i] = GetPar(inFile);
      }
      for (int i = 0; i < DEGx2; i++) {
        this->tx00[bi + i] = 1e-3f * GetPar(inFile);
      }
      bi = UttOffset(ix, iy, DEGx1);
      for (int i = 0; i < DEGx1; i++) {
        this->x10[bi + i] = GetPar(inFile);
      }
      for (int i = 0; i < DEGx1; i++) {
        this->x01[bi + i] = GetPar(inFile);
      }
      for (int i = 0; i < DEGx1; i++) {
        this->tx10[bi + i] = 1e-3f * GetPar(inFile);
      }
      for (int i = 0; i < DEGx1; i++) {
        this->tx01[bi + i] = 1e-3f * GetPar(inFile);
      }
      bi = UttOffset(ix, iy, DEGy2);
      for (int i = 0; i < DEGy2; i++) {
        this->y00[bi + i] = GetPar(inFile);
      }
      for (int i = 0; i < DEGy2; i++) {
        this->ty00[bi + i] = 1e-3f * GetPar(inFile);
      }
      bi = UttOffset(ix, iy, DEGy1);
      for (int i = 0; i < DEGy1; i++) {
        this->y10[bi + i] = GetPar(inFile);
      }
      for (int i = 0; i < DEGy1; i++) {
        this->y01[bi + i] = GetPar(inFile);
      }
      for (int i = 0; i < DEGy1; i++) {
        this->ty10[bi + i] = 1e-3f * GetPar(inFile);
      }
      for (int i = 0; i < DEGy1; i++) {
        this->ty01[bi + i] = 1e-3f * GetPar(inFile);
      }
      if (inFile.eof()) return 0;
      return 1;
    }
  }; // End KalmanParametrizations
} // namespace ParKalmanFilter

namespace ParKalmanFilter {

  ////////////////////////////////////////////////////////////////////////
  // Read parameters.
  ////////////////////////////////////////////////////////////////////////

  //----------------------------------------------------------------------
  //  read parameters from file - here for the extrapolation UT -> T
  __host__ void KalmanParametrizations::read_params_UTT(std::string file)
  {
    std::string line;
    std::ifstream myfile(file);
    if (!myfile.is_open()) throw StrException("Failed to open parameter file.");
    myfile >> ZINI >> ZFIN >> PMIN >> BENDX >> BENDX_X2 >> BENDX_Y2 >> BENDY_XY >> Txmax >> Tymax >> XFmax >> Dtxy;
    myfile >> Nbinx >> Nbiny >> XGridOption >> YGridOption >> DEGX1 >> DEGX2 >> DEGY1 >> DEGY2;
    for (int ix = 0; ix < Nbinx; ix++) {
      for (int iy = 0; iy < Nbiny; iy++) {
        int ec = Read(ix, iy, myfile);
        if (ec == 0)
          throw StrException("Something went wrong in reading the Parameter file. Are your ParamFiles up to date?");
      }
    }
    Xmax = ZINI * Txmax;
    Ymax = ZINI * Tymax;
  }

  __device__ __host__ KalmanFloat KalmanParametrizations::UTTExtrEndZ() const { return ZFIN; }

  __device__ __host__ KalmanFloat KalmanParametrizations::UTTExtrBeginZ() const { return ZINI; }

} // namespace ParKalmanFilter
