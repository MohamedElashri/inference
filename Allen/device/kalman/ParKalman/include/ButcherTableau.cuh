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
    a_table(1.) b_table(1., 0.) b_star_table(static_cast<ftype>(1. / 2.), static_cast<ftype>(1. / 2.))
  };

  template<typename ftype = float>
  struct RK4 {
    static constexpr int N_stages = 4;
    a_table(
      static_cast<ftype>(1. / 2.),
      static_cast<ftype>(0.),
      static_cast<ftype>(1. / 2.),
      static_cast<ftype>(0.),
      static_cast<ftype>(0.),
      static_cast<ftype>(1))
      b_table(
        static_cast<ftype>(1. / 6.),
        static_cast<ftype>(1. / 3.),
        static_cast<ftype>(1. / 3.),
        static_cast<ftype>(1. / 6.))
  };

  template<typename ftype = float>
  struct CashKarp {
    static constexpr int N_stages = 6;
    a_table(
      static_cast<ftype>(1. / 5.), //
      static_cast<ftype>(3. / 40.),
      static_cast<ftype>(9. / 40.), //
      static_cast<ftype>(3. / 10.),
      static_cast<ftype>(-9. / 10.),
      static_cast<ftype>(6. / 5.), //
      static_cast<ftype>(-11. / 54.),
      static_cast<ftype>(5. / 2.),
      static_cast<ftype>(-70. / 27.),
      static_cast<ftype>(35. / 27.), //
      static_cast<ftype>(1631. / 55296.),
      static_cast<ftype>(175. / 512.),
      static_cast<ftype>(575. / 13824.),
      static_cast<ftype>(44275. / 110592.),
      static_cast<ftype>(253. / 4096.))
      b_table(
        static_cast<ftype>(37. / 378.),
        static_cast<ftype>(0.),
        static_cast<ftype>(250. / 621.),
        static_cast<ftype>(125. / 594.),
        static_cast<ftype>(0.),
        static_cast<ftype>(512. / 1771.))
        b_star_table(
          static_cast<ftype>(2825. / 27648.),
          static_cast<ftype>(0.),
          static_cast<ftype>(18575. / 48384.),
          static_cast<ftype>(13525. / 55296.),
          static_cast<ftype>(277. / 14336.),
          static_cast<ftype>(1. / 4.))
  };
} // namespace ButcherTableau
