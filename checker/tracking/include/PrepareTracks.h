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
#pragma once

#include <vector>
#include "TrackChecker.h"
#include "Logger.h"
#include "UTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "UTEventModel.cuh"
#include "States.cuh"
#include "Datatype.cuh"
#include "ParticleTypes.cuh"

/**
 * @brief Prepares tracks for Velo consolidated datatypes.
 */
std::vector<Checker::Tracks> prepareVeloTracks(
  const unsigned number_of_events,
  std::span<const unsigned> track_atomics,
  std::span<const unsigned> track_hit_number,
  std::span<const char> track_hits,
  std::span<const mask_t> event_list);

/**
 * @brief Prepares tracks for Velo, UT consolidated datatypes.
 */
std::vector<Checker::Tracks> prepareUTTracks(
  const unsigned number_of_events,
  std::span<const unsigned> velo_track_atomics,
  std::span<const unsigned> velo_track_hit_number,
  std::span<const char> velo_track_hits,
  std::span<const char> kalman_velo_states,
  std::span<const unsigned> ut_track_atomics,
  std::span<const unsigned> ut_track_hit_number,
  std::span<const char> ut_track_hits,
  std::span<const unsigned> ut_track_velo_indices,
  std::span<const float> ut_qop,
  std::span<const mask_t> event_list);

std::vector<Checker::Tracks> prepareUnmatchedSeedingTracks(
  const unsigned number_of_events,
  std::span<const bool> matched_is_scifi_track_used,
  std::span<const unsigned> scifi_seed_atomics,
  std::span<const unsigned> scifi_seed_hit_number,
  std::span<const char> scifi_seed_hits,
  std::span<const SciFi::Seeding::Track> scifi_seeds,
  std::span<const MiniState> seeding_states,
  std::span<const mask_t> event_list);

std::vector<Checker::Tracks> prepareSeedingTracks(
  const unsigned number_of_events,
  std::span<const unsigned> scifi_seed_atomics,
  std::span<const unsigned> scifi_seed_hit_number,
  std::span<const char> scifi_seed_hits,
  std::span<const SciFi::Seeding::Track> scifi_seeds,
  std::span<const MiniState> seeding_states,
  std::span<const mask_t> event_list);

std::vector<Checker::Tracks> prepareSeedingTracksXZ(
  const unsigned number_of_events,
  std::span<const unsigned> scifi_seed_atomics,
  std::span<const unsigned> scifi_seed_hit_number,
  std::span<const char> scifi_hits,
  std::span<const unsigned> scifi_hit_count,
  std::span<const SciFi::Seeding::TrackXZ> scifi_seeds,
  std::span<const mask_t> event_list);

/**
 * @brief Read forward tracks from binary files
 */
std::vector<Checker::Tracks> read_forward_tracks(const char* events, const unsigned* event_offsets, const int n_events);
