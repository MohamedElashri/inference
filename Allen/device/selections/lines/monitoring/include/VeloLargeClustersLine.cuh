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

#include "AlgorithmTypes.cuh"
#include "Line.cuh"
#include "VeloConsolidated.cuh"
#include "ODINBank.cuh"

namespace velo_large_clusters_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_velo_states_view_t, Allen::Views::Physics::KalmanStates) dev_velo_states_view;
    DEVICE_INPUT(dev_offsets_velo_tracks_t, unsigned) dev_offsets_velo_tracks;
    HOST_OUTPUT(host_line_data_t, LineData) host_line_data;
    HOST_OUTPUT(host_fn_parameters_t, char) host_fn_parameters;
  };

  struct velo_large_clusters_line_t : public SelectionAlgorithm,
                                      Parameters,
                                      Line<velo_large_clusters_line_t, Parameters> {
    struct DeviceProperties {
      unsigned min_cluster_size;
      float min_eta;
      unsigned min_n_hits;
      DeviceProperties(const velo_large_clusters_line_t& algo, const Allen::Context&) :
        min_cluster_size(algo.m_min_cluster_size), min_eta(algo.m_min_eta), min_n_hits(algo.m_min_n_hits)
      {}
    };
    __device__ static std::tuple<const float, const Allen::Views::Velo::Consolidated::Track>
    get_input(const Parameters& parameters, const unsigned event_number, const unsigned i);

    __device__ static bool select(
      const Parameters&,
      const DeviceProperties&,
      std::tuple<const float, const Allen::Views::Velo::Consolidated::Track> input);

    static unsigned get_decisions_size(const ArgumentReferences<Parameters>& arguments)
    {
      return first<typename Parameters::host_number_of_reconstructed_velo_tracks_t>(arguments);
    }

    __device__ static unsigned offset(const Parameters& parameters, const unsigned event_number)
    {
      return parameters.dev_offsets_velo_tracks[event_number];
    }

  private:
    Allen::Property<unsigned> m_min_cluster_size {
      this,
      "min_cluster_size",
      4,
      "Minimum cluster size for accepted hits"};
    Allen::Property<float> m_min_eta {this, "min_eta", 5.f, "Minimum eta for accepted tracks"};
    Allen::Property<unsigned> m_min_n_hits {this, "min_n_hits", 3, "min number of large clusters in velo track"};
  };
} // namespace velo_large_clusters_line
