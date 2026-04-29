/*****************************************************************************\
* (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "MEPTools.h"

namespace Codex {

  // Defined by hardware/firmware
  constexpr unsigned NumberOfPhiStrips = 64;
  constexpr unsigned NumberOfEtaStrips = 32;
  constexpr unsigned NumberOfDCTs = 14; // each DCT corresponds to a triplet of singlets, so 14 DCTs for 42 singlets
  constexpr unsigned NumberOfSinglets = 42;
  constexpr unsigned MaxStripId = 143; //

  // Time window in clock cycles (30 clock cycles per event), used for clustering and coincidences
  constexpr float TimeWindow = 15; // half event

  // Distance Window for coincidence matching in strip units
  constexpr float DistanceWindow = 10; // to be tuned, currently set to a high value

  // Maximum sizes, set to safe values
  constexpr unsigned MaxHitsPerEvent = 512; // Maximum occupancy per event
  constexpr unsigned MaxPhiClustersPerSinglet =
    64; // Maximum number of phi clusters per singlet, set to total number of phi strips
  constexpr unsigned MaxEtaClustersPerSinglet =
    32; // Maximum number of eta clusters per singlet, set to total number of eta strips
  constexpr unsigned MaxEndClustersPerSinglet = 64; // Maximum number of channel clusters (combinations of phi and eta
                                                    // clusters) per singlet
  constexpr unsigned MaxCoincidencePerTriplet = 64; // Maximum number of coincidences per triplet

  struct RawBank {
    uint32_t source_id = 0;
    uint8_t const* data = nullptr;
    uint8_t const* end = nullptr;
    uint8_t const type;

    // For Allen format
    __device__ __host__ RawBank(const char* raw_bank, const uint16_t s, const uint8_t t) :
      RawBank {*reinterpret_cast<uint32_t const*>(raw_bank), raw_bank + sizeof(uint32_t), s, t}
    {}

    // For MEP format
    __device__ __host__ RawBank(const uint32_t sid, const char* fragment, const uint16_t s, const uint8_t t) :
      source_id {sid}, data {reinterpret_cast<uint8_t const*>(fragment)},
      end {reinterpret_cast<uint8_t const*>(fragment + s)}, type {t}
    {}
  };

  template<bool mep_layout>
  struct RawEvent {

    uint32_t number_of_raw_banks = 0;
    const char* data = nullptr;
    const uint32_t* offsets = nullptr;
    typename std::conditional_t<mep_layout, uint32_t const, uint16_t const>* sizes = nullptr;
    typename std::conditional_t<mep_layout, uint32_t const, uint8_t const>* types = nullptr;
    const unsigned event = 0;

    // For Allen format
    __device__ __host__
    RawEvent(char const* d, uint32_t const* o, uint32_t const* s, uint32_t const* t, unsigned const event_number) :
      offsets {o},
      event {event_number}
    {
      if constexpr (mep_layout) {
        data = d;
        number_of_raw_banks = MEP::number_of_banks(o);
        sizes = s;
        types = t;
      }
      else {
        data = d + offsets[event];
        number_of_raw_banks = reinterpret_cast<uint32_t const*>(data)[0];
        sizes = Allen::bank_sizes(s, event);
        types = Allen::bank_types(t, event);
      }
    }

    __device__ __host__ RawBank raw_bank(unsigned const n) const
    {
      if constexpr (mep_layout) {
        return MEP::raw_bank<RawBank>(data, offsets, sizes, types, event, n);
      }
      else {
        uint32_t const* bank_offsets = reinterpret_cast<uint32_t const*>(data) + 1;
        return RawBank {data + (number_of_raw_banks + 2) * sizeof(uint32_t) + bank_offsets[n], sizes[n], types[n]};
      }
    }
  };
} // namespace Codex

struct CodexHit {
  uint8_t singlet_id;
  uint8_t strip_id;
  uint8_t strip_type;
  uint8_t time;

  CodexHit() = default;
  __device__ CodexHit(uint8_t singlet, uint8_t strip, uint8_t type, uint8_t t) :
    singlet_id(singlet), strip_id(strip), strip_type(type), time(t)
  {}
};

struct CodexSideCluster {
  unsigned strips_size;

  int strip_ids_sum;

  int cluster_times_sum;

  int singlet_id;

  int strip_type;

  CodexSideCluster() = default;

  __device__ CodexSideCluster(uint8_t strip_id, uint8_t hit_time, uint8_t singlet_id, uint8_t strip_type) :
    strips_size(1u), strip_ids_sum((int) strip_id), cluster_times_sum((int) hit_time), singlet_id((int) singlet_id),
    strip_type((int) strip_type)
  {}

  __device__ float get_cluster_time() const
  {
    return static_cast<float>(cluster_times_sum) / static_cast<float>(strips_size);
  }

  __device__ int get_cluster_strip_id() const
  {
    return static_cast<int>((strip_ids_sum + strips_size / 2) / strips_size);
  }

  __device__ void addHit(uint8_t strip_id, uint8_t hit_time)
  {
    strips_size++;
    strip_ids_sum = strip_ids_sum + (int) strip_id;
    cluster_times_sum = cluster_times_sum + (int) hit_time;
  }
};

struct CodexCluster {
  size_t eta_size;
  size_t phi_size;

  int phi_strip_mean;
  int eta_strip_mean;

  float cluster_mean_time;
  int singlet_id;

  CodexCluster() = default;
  __device__
  CodexCluster(int eta_size, int phi_size, int phi_strip_mean, int eta_strip_mean, float cluster_time, int singlet_id) :
    eta_size(eta_size),
    phi_size(phi_size), phi_strip_mean(phi_strip_mean), eta_strip_mean(eta_strip_mean), cluster_mean_time(cluster_time),
    singlet_id(singlet_id)
  {}
};

struct CodexCoincidence {
  int coincidence_type;

  int phi_strip_mean;
  int eta_strip_mean;
  float mean_time;

  unsigned layer_0_clusted_index {32768}; //  used as a high default value in case of hit not found,
                                          //  MaxEndClustersPerSinglet = 2048 so 32768 is impossible to reach
  unsigned layer_1_clusted_index {32768};
  unsigned layer_2_clusted_index {32768};

  CodexCoincidence() = default;
  __device__
  CodexCoincidence(int coincidence_type, int phi, int eta, float time, unsigned l0_cl_id, unsigned l1_cl_id) :
    coincidence_type(coincidence_type),
    phi_strip_mean(phi), eta_strip_mean(eta), mean_time(time)
  {
    if (coincidence_type == 1) {
      // layer 0 ↔ layer 1
      layer_0_clusted_index = l0_cl_id;
      layer_1_clusted_index = l1_cl_id;
    }
    else if (coincidence_type == 2) {
      // layer 1 ↔ layer 2
      layer_1_clusted_index = l0_cl_id;
      layer_2_clusted_index = l1_cl_id;
    }
    else if (coincidence_type == 3) {
      // layer 0 ↔ layer 2
      layer_0_clusted_index = l0_cl_id;
      layer_2_clusted_index = l1_cl_id;
    }
  }

  __device__ void update_coincidence(int phi, int eta, float time, int l_id)
  {
    // we do not update the phi mean and eta mean, since we are not going to use them after
    coincidence_type = 0;
    layer_2_clusted_index = l_id;

    // mean value update used only for validation purposes
    phi_strip_mean = (phi_strip_mean * 2 + phi) / 3;
    eta_strip_mean = (eta_strip_mean * 2 + eta) / 3;
    mean_time = (mean_time * 2 + time) / 3;
  }
};