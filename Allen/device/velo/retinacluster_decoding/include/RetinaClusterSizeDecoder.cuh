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

#include <array>
#include <cstdint>

namespace decode_retinaclusters {

  // ============================================================================
  // VELO Retina cluster size lookup tables
  //
  // Tables are built at compile time, replicating the logic from
  // LHCb/VP/VPDAQ/src/VPRetinaTopologyID.cpp
  // ============================================================================

  namespace size_table_detail {

    constexpr int popcount8(uint32_t x)
    {
      int n = 0;
      while (x) {
        n += x & 1;
        x >>= 1;
      }
      return n;
    }

    // ----------------------------------------------------------------
    // 2x4 super-pixel topology (isolated clusters)
    // ----------------------------------------------------------------

    struct Sub2x4 {
      uint32_t sp0;
      uint32_t sp1;
    };

    constexpr Sub2x4 breakdown_topo2x4(uint32_t topo)
    {
      uint32_t sp0 = 0, sp1 = 0;
      if ((topo & 2) == 0 && (topo & 32) == 0) {
        sp0 = topo & (1 + 16);
        sp1 = topo & (4 + 8 + 64 + 128);
      }
      else if ((topo & 4) == 0 && (topo & 64) == 0) {
        sp0 = topo & (1 + 2 + 16 + 32);
        sp1 = topo & (8 + 128);
      }
      else {
        sp0 = topo;
        sp1 = 0;
      }
      if (sp0 == 0) return {sp1, 0};
      return {sp0, sp1};
    }

    struct Frac {
      uint32_t row;
      uint32_t col;
    };

    constexpr Frac frac2x4cluster(uint32_t topo)
    {
      uint32_t shift_col = 0, shift_row = 0, n = 0;
      for (uint32_t ix = 0; ix < 2; ++ix) {
        for (uint32_t iy = 0; iy < 4; ++iy) {
          if (topo & (1u << (ix * 4 + iy))) {
            shift_col += ix;
            shift_row += iy;
            ++n;
          }
        }
      }
      if (n == 0) return {0, 0};
      return {((shift_row * 4) + n / 2) / n % 4, ((shift_col * 4) + n / 2) / n % 4};
    }

    constexpr std::array<uint32_t, 256> topo2x4_to_topoIDfrac()
    {
      std::array<uint32_t, 256> out {};
      std::array<uint32_t, 256> vec_topoID {};
      std::array<uint32_t, 256> fr0 {}, fc0 {}, fr1 {}, fc1 {};
      std::array<uint32_t, 16> cnt {};

      for (uint32_t topo = 0; topo < 256; ++topo) {
        if (topo == 0) {
          vec_topoID[topo] = (63u << 6) + 63u;
          continue;
        }
        const auto sub = breakdown_topo2x4(topo);
        const uint32_t sp0 = sub.sp0;
        const uint32_t sp1 = sub.sp1;
        uint32_t id0 = 0, id1 = 0;

        if (sp0 == 0) {
          id0 = 63;
        }
        else {
          const auto f = frac2x4cluster(sp0);
          fr0[topo] = f.row;
          fc0[topo] = f.col;
          if (sp1 == 0) {
            id0 = cnt[fr0[topo] * 4 + fc0[topo]]++;
          }
          else {
            id0 = vec_topoID[sp0] & 63;
          }
        }
        if (sp1 == 0) {
          id1 = 63;
        }
        else {
          const auto f = frac2x4cluster(sp1);
          fr1[topo] = f.row;
          fc1[topo] = f.col;
          if (sp0 == 0) {
            id1 = cnt[fr1[topo] * 4 + fc1[topo]]++;
          }
          else {
            id1 = vec_topoID[sp1] & 63;
          }
        }
        vec_topoID[topo] = (id1 << 6) + id0;
      }

      for (uint32_t topo = 0; topo < 256; ++topo) {
        out[topo] = (((vec_topoID[topo] >> 6) & 63) << 14) | (fc1[topo] << 12) | (fr1[topo] << 10) |
                    ((vec_topoID[topo] & 63) << 4) | (fc0[topo] << 2) | fr0[topo];
      }
      return out;
    }

    // ----------------------------------------------------------------
    // 3x3 pixel topology (non-isolated clusters)
    // ----------------------------------------------------------------

    constexpr bool is3x3cluster(uint32_t t) { return (t & 1) || ((t & 2) && (t & 8)); }

    constexpr bool is3x3revcluster(uint32_t t) { return (t & 64) || ((t & 8) && (t & 128)); }

    constexpr uint32_t equivalent_topo3x3(uint32_t t)
    {
      if (!(t & 2) && !(t & 8) && !(t & 16)) return t & 1;
      if (!(t & 8) && !(t & 16) && !(t & 32)) return t & (1 + 2 + 4);
      if (!(t & 2) && !(t & 16) && !(t & 128)) return t & (1 + 8 + 64);
      if (!(t & 16) && !(t & 32) && !(t & 128)) return t & (1 + 2 + 4 + 8 + 64);
      if (!(t & 2) && !(t & 16) && !(t & 32)) return t & (1 + 8 + 64 + 128 + 256);
      if (!(t & 8) && !(t & 16) && !(t & 128)) return t & (1 + 2 + 4 + 32 + 256);
      return t;
    }

    constexpr uint32_t equivalent_topo3x3_rev(uint32_t t)
    {
      if (!(t & 8) && !(t & 16) && !(t & 128)) return t & 0x40;
      if (!(t & 8) && !(t & 16) && !(t & 32)) return t & (0x40 + 0x80 + 0x100);
      if (!(t & 2) && !(t & 16) && !(t & 128)) return t & (0x1 + 0x8 + 0x40);
      if (!(t & 16) && !(t & 32) && !(t & 128)) return t & (0x1 + 0x2 + 0x4 + 0x8 + 0x40);
      if (!(t & 2) && !(t & 16) && !(t & 32)) return t & (0x1 + 0x8 + 0x40 + 0x80 + 0x100);
      if (!(t & 2) && !(t & 8) && !(t & 16)) return t & (0x4 + 0x20 + 0x40 + 0x80 + 0x100);
      return t;
    }

    constexpr Frac frac3x3cluster(uint32_t topo)
    {
      uint32_t shift_col = 0, shift_row = 0, n = 0;
      for (uint32_t ix = 0; ix < 3; ++ix) {
        for (uint32_t iy = 0; iy < 3; ++iy) {
          if (topo & (1u << (ix * 3 + iy))) {
            shift_col += ix;
            shift_row += iy;
            ++n;
          }
        }
      }
      if (n == 0) return {0, 0};
      return {((shift_row * 4) + n / 2) / n % 4, ((shift_col * 4) + n / 2) / n % 4};
    }

    template<bool Reverse>
    constexpr std::array<uint32_t, 512> topo3x3_to_topoIDfrac()
    {
      std::array<uint32_t, 512> out {};
      std::array<uint32_t, 512> vec_topoID {};
      std::array<uint32_t, 512> fr {}, fc {};
      std::array<uint32_t, 16> cnt {};

      for (uint32_t topo = 0; topo < 512; ++topo) {
        const bool valid = Reverse ? is3x3revcluster(topo) : is3x3cluster(topo);
        if (!valid) {
          vec_topoID[topo] = 31;
          continue;
        }
        const uint32_t eq = Reverse ? equivalent_topo3x3_rev(topo) : equivalent_topo3x3(topo);
        const auto f = frac3x3cluster(eq);
        fr[topo] = f.row;
        fc[topo] = f.col;
        if (topo == eq) {
          vec_topoID[topo] = cnt[fr[topo] * 4 + fc[topo]]++;
        }
        else {
          vec_topoID[topo] = vec_topoID[eq];
        }
      }
      for (uint32_t topo = 0; topo < 512; ++topo) {
        out[topo] = (vec_topoID[topo] << 4) | (fc[topo] << 2) | fr[topo];
      }
      return out;
    }

  } // namespace size_table_detail

  // ============================================================================
  // The three size lookup tables, built at compile time.
  //
  // Key layout: (topoID << 4) | (fx << 2) | fy
  //   - h_velo_cluster_size_iso       : isolated, topoID 6-bit, 1024 entries
  //   - h_velo_cluster_size_noniso    : non-isolated direct (sensor%4==0,3), 512
  //   - h_velo_cluster_size_noniso_rev: non-isolated reverse (sensor%4==1,2), 512
  //
  // ============================================================================

  // Note on the initial fill value:
  //   We pre-fill every slot with 1 instead of 0. Valid (topoID, fx, fy)
  //   combinations are then overwritten with their actual popcount. Unfilled
  //   slots therefore keep the value 1, which mirrors the behaviour of
  //   VPRetinaFullClusterDecoder's "hot-fix for cluster with no pixels" branch
  //   (see LHCb/VP/VPDAQ/src/VPRetinaFullClusterDecoder.cpp), where invalid
  //   topology lookups fall back to a single placeholder pixel.

  constexpr auto h_velo_cluster_size_iso = []() {
    std::array<uint8_t, 1024> table {};
    for (auto& v : table)
      v = 1;
    const auto fwd = size_table_detail::topo2x4_to_topoIDfrac();
    for (uint32_t sp = 1; sp < 256; ++sp) {
      const auto sub = size_table_detail::breakdown_topo2x4(sp);
      if (sub.sp0 != 0) {
        const uint32_t key0 = fwd[sp] & 0x3FF;
        table[key0] = static_cast<uint8_t>(size_table_detail::popcount8(sub.sp0));
      }
      if (sub.sp1 != 0) {
        const uint32_t key1 = (fwd[sp] >> 10) & 0x3FF;
        table[key1] = static_cast<uint8_t>(size_table_detail::popcount8(sub.sp1));
      }
    }
    return table;
  }();

  constexpr auto h_velo_cluster_size_noniso = []() {
    std::array<uint8_t, 512> table {};
    for (auto& v : table)
      v = 1;
    const auto fwd = size_table_detail::topo3x3_to_topoIDfrac<false>();
    for (uint32_t topo = 0; topo < 512; ++topo) {
      if (!size_table_detail::is3x3cluster(topo)) continue;
      const uint32_t eq = size_table_detail::equivalent_topo3x3(topo);
      const uint32_t key = fwd[topo] & 0x1FF;
      table[key] = static_cast<uint8_t>(size_table_detail::popcount8(eq));
    }
    return table;
  }();

  constexpr auto h_velo_cluster_size_noniso_rev = []() {
    std::array<uint8_t, 512> table {};
    for (auto& v : table)
      v = 1;
    const auto fwd = size_table_detail::topo3x3_to_topoIDfrac<true>();
    for (uint32_t topo = 0; topo < 512; ++topo) {
      if (!size_table_detail::is3x3revcluster(topo)) continue;
      const uint32_t eq = size_table_detail::equivalent_topo3x3_rev(topo);
      const uint32_t key = fwd[topo] & 0x1FF;
      table[key] = static_cast<uint8_t>(size_table_detail::popcount8(eq));
    }
    return table;
  }();

} // namespace decode_retinaclusters
