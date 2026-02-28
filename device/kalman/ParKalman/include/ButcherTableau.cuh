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

// This is a workaround to cuda not allowing static constexpr data member on device
// Hidden in macros to make the definitions more readable
// It checks the number of values match N_stages
#define a_table(...)                                                                                                 \
  __device__ static constexpr ftype a(int i, int j)                                                                  \
  {                                                                                                                  \
    ftype a[] {__VA_ARGS__};                                                                                         \
    static_assert(N_stages * (N_stages - 1) / 2 == 0 || N_stages * (N_stages - 1) / 2 == sizeof(a) / sizeof(ftype)); \
    return a[i * (i - 1) / 2 + j];                                                                                   \
  }
#define b_table(...)                         \
  __device__ static constexpr ftype b(int i) \
  {                                          \
    ftype b[N_stages] {__VA_ARGS__};         \
    return b[i];                             \
  }
#define b_star_table(...)                         \
  __device__ static constexpr ftype b_star(int i) \
  {                                               \
    ftype b_star[N_stages] {__VA_ARGS__};         \
    return b_star[i];                             \
  }

namespace ButcherTableau {
  // c1 |
  // c2 | a1
  // c3 | a2 a3
  // .. | a4 a5 a6
  // cs | ...
  // ------------------
  //    | b1 b2 B3 ... bs

  template<typename ftype = float>
  struct Euler {
    static constexpr int N_stages = 1;
    a_table(0.) b_table(1.)
  };

  template<typename ftype = float>
  struct HeunEuler {
    static constexpr int N_stages = 2;
    a_table(1.) b_table(1., 0.) b_star_table(1. / 2., 1. / 2.)
  };

  template<typename ftype = float>
  struct RK4 {
    static constexpr int N_stages = 4;
    a_table(
      1. / 2., //
      0.,
      1. / 2., //
      0.,
      0.,
      1. //
      ) b_table(1. / 6., 1. / 3., 1. / 3., 1. / 6.)
  };

  template<typename ftype = float>
  struct CashKarp {
    static constexpr int N_stages = 6;
    a_table(
      1. / 5., //
      3. / 40.,
      9. / 40., //
      3. / 10.,
      -9. / 10.,
      6. / 5., //
      -11. / 54.,
      5. / 2.,
      -70. / 27.,
      35. / 27., //
      1631. / 55296.,
      175. / 512.,
      575. / 13824.,
      44275. / 110592.,
      253. / 4096.) b_table(37. / 378., 0., 250. / 621., 125. / 594., 0., 512. / 1771.)
      b_star_table(2825. / 27648., 0., 18575. / 48384., 13525. / 55296., 277. / 14336., 1. / 4.)
  };
} // namespace ButcherTableau
