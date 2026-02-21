/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration          *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

// Basic
#include "AlgorithmTypes.cuh"
// Common
#include "States.cuh"
#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "ParticleTypes.cuh"

namespace filter_ttracks {

  struct Parameters {
    // Basic
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    // Long track VETO
    DEVICE_INPUT(dev_matched_is_scifi_track_used_t, bool) dev_matched_is_scifi_track_used;
    // Downstream track VETO
    DEVICE_INPUT(dev_multi_event_downstream_tracks_view_t, Allen::Views::Physics::MultiEventDownstreamTracks)
    dev_multi_event_downstream_tracks_view;

    // SciFi input
    HOST_INPUT(host_number_of_scifi_tracks_t, unsigned) host_number_of_scifi_tracks;
    DEVICE_INPUT(dev_seeding_qop_t, float) dev_seeding_qop;
    DEVICE_INPUT(dev_seeding_states_t, MiniState) dev_seeding_states;
    DEVICE_INPUT(dev_offsets_seeding_tracks_t, unsigned) dev_offsets_seeding_tracks;
    DEVICE_INPUT(dev_seeding_tracks_t, SciFi::Seeding::Track) dev_seeding_tracks;
    DEVICE_INPUT(dev_scifi_track_view_t, Allen::Views::SciFi::Consolidated::Track)
    dev_scifi_track_view;

    // Filtered TT tracks output
    HOST_OUTPUT(host_number_of_filtered_scifi_tracks_t, unsigned) host_number_of_filtered_scifi_tracks;
    DEVICE_OUTPUT(dev_offsets_filtered_seeding_tracks_t, unsigned) dev_offsets_filtered_seeding_tracks;
    DEVICE_OUTPUT(dev_filtered_seeding_qop_t, float) dev_filtered_seeding_qop;
    DEVICE_OUTPUT(dev_filtered_seeding_states_t, float) dev_filtered_seeding_states;
    DEVICE_OUTPUT(dev_filtered_state_indexes_t, unsigned) dev_filtered_state_indexes;

    // Just because we need MiniState for the is_muon_t algorithm...
    DEVICE_OUTPUT(dev_muon_states_t, MiniState) dev_muon_states;

    // Buffer
    DEVICE_OUTPUT(dev_buffer_filtered_state_indexes_t, unsigned) dev_buffer_filtered_state_indexes;
    DEVICE_OUTPUT(dev_buffer_filtered_seeding_qop_t, float) dev_buffer_filtered_seeding_qop;
    DEVICE_OUTPUT(dev_buffer_filtered_seeding_states_t, float) dev_buffer_filtered_seeding_states;
    DEVICE_OUTPUT(dev_buffer_used_tracks_t, bool) dev_buffer_used_tracks;
  };

  struct filter_ttracks_t : public DeviceAlgorithm, Parameters {
    void update(const Constants& constants) const;

    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants& constants,
      const Allen::Context& context) const;

    struct DeviceProperties {
      float max_abs_qop;
      float max_clonekiller_dx;
      float max_clonekiller_dy;
      float max_clonekiller_dtx;
      float max_clonekiller_dty;
      short max_clonekiller_allowed_shared_hits;

      DeviceProperties(const filter_ttracks_t& algo) :
        max_abs_qop(algo.m_max_abs_qop.value()), max_clonekiller_dx(algo.m_max_clonekiller_dx.value()),
        max_clonekiller_dy(algo.m_max_clonekiller_dy.value()), max_clonekiller_dtx(algo.m_max_clonekiller_dtx.value()),
        max_clonekiller_dty(algo.m_max_clonekiller_dty.value()),
        max_clonekiller_allowed_shared_hits(algo.m_max_clonekiller_allowed_shared_hits.value())
      {}
    };

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {128, 1, 1}, "block dimensions"};

    Allen::Property<float> m_max_abs_qop {this,
                                          "max_abs_qop",
                                          1.f / 5000.f,
                                          "maximum absolute value of qop (in 1/MeV)"};

    Allen::Property<float> m_max_clonekiller_dx {this,
                                                 "max_clonekiller_dx",
                                                 1.f,
                                                 "maximum delta x between tracks to run clone killer (in mm)"};

    Allen::Property<float> m_max_clonekiller_dy {this,
                                                 "max_clonekiller_dy",
                                                 15.f,
                                                 "maximum delta y between tracks to run clone killer (in mm)"};

    Allen::Property<float> m_max_clonekiller_dtx {this,
                                                  "max_clonekiller_dtx",
                                                  0.008f,
                                                  "maximum delta tx between tracks to run clone killer (in mm)"};

    Allen::Property<float> m_max_clonekiller_dty {this,
                                                  "max_clonekiller_dty",
                                                  0.015f,
                                                  "maximum delta ty between tracks to run clone killer (in mm)"};

    Allen::Property<short> m_max_clonekiller_allowed_shared_hits {
      this,
      "max_clonekiller_allowed_shared_hits",
      7,
      "maximum allowed number of shared hits between tracks"};

    Allen::Property<bool> m_with_kalman_filter {this,
                                                "with_kalman_filter",
                                                true,
                                                "to enable the kalman filter on the filtered tracks"};
  };

  __constant__ float dev_average_dxdy[SciFi::Constants::n_layers];

  __global__ void filter_ttracks(Parameters, float, const filter_ttracks_t::DeviceProperties);
  __global__ void copy_ttracks(Parameters);
  __global__ void kalman_filter(Parameters, const MagneticField::Magfield field);
  __global__ void fill_muon_states(Parameters, const MagneticField::Magfield field);
} // namespace filter_ttracks
