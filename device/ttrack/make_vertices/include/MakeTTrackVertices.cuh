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
#include "RungeKuttaExtrapolator.cuh"
#include "VertexDefinitions.cuh"
#include "TTrackDefinitions.cuh"

#include "AllenMonitoring.h"

namespace make_ttrack_vertices {

  struct Parameters {
    Allen::KernelInvocationConfiguration config;
    // Basic
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    // Filtered TTrack input
    DEVICE_INPUT(dev_offsets_seeding_tracks_t, unsigned) dev_offsets_seeding_tracks;
    DEVICE_INPUT(dev_filtered_ttracks_qop_t, float) dev_filtered_ttracks_qop;
    DEVICE_INPUT(dev_filtered_ttracks_states_t, float) dev_filtered_ttracks_states;

    DEVICE_OUTPUT(dev_tt_vertices_t, float) dev_tt_vertices;
    DEVICE_OUTPUT(dev_offsets_tt_vertices_t, unsigned) dev_offsets_tt_vertices;
    HOST_OUTPUT(host_number_of_tt_vertices_t, unsigned) host_number_of_tt_vertices;
  };

  struct make_ttrack_vertices_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants& constants,
      const Allen::Context& context) const;

    struct DeviceProperties {
      float min_vrtx_z;
      float max_vrtx_z;
      float min_yzinter_z;
      float max_yzinter_z;
      float max_doca;
      float min_distance_to_beamline;
      float min_distance_to_beamline_slope;
      float min_p2;
      float min_pt2;

      Allen::Monitoring::Histogram2D<>::DeviceType histogram_xz_vrtx_pos;
      Allen::Monitoring::Histogram2D<>::DeviceType histogram_yz_vrtx_pos;

      DeviceProperties(const make_ttrack_vertices_t& algo, const Allen::Context& ctx) :
        min_vrtx_z(algo.m_min_vrtx_z.value()), max_vrtx_z(algo.m_max_vrtx_z.value()),
        min_yzinter_z(algo.m_min_yzinter_z.value()), max_yzinter_z(algo.m_max_yzinter_z.value()),
        max_doca(algo.m_max_doca.value()), min_distance_to_beamline(algo.m_min_distance_to_beamline.value()),
        min_distance_to_beamline_slope(algo.m_min_distance_to_beamline_slope.value()), min_p2(algo.m_min_p2.value()),
        min_pt2(algo.m_min_pt2.value()), histogram_xz_vrtx_pos(algo.m_xz_vrtx_pos.data(ctx)),
        histogram_yz_vrtx_pos(algo.m_yz_vrtx_pos.data(ctx))
      {}
    };

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};

    Allen::Property<float> m_min_yzinter_z {
      this,
      "min_yzinter_z",
      1500.0,
      "minimum z-coordinate of y-z intersection (in mm)"};
    Allen::Property<float> m_max_yzinter_z {
      this,
      "max_yzinter_z",
      9500.0,
      "maximum z-coordinate of y-z intersection (in mm)"};

    Allen::Property<float> m_min_vrtx_z {this, "min_vrtx_z", 2000.0, "minimum z-coordinate of vertices (in mm)"};
    Allen::Property<float> m_max_vrtx_z {this, "max_vrtx_z", 9000.0, "maximum z-coordinate of vertices (in mm)"};
    Allen::Property<float> m_max_doca {this, "max_doca", 40.0, "maximum distance of closest approach (in mm)"};
    Allen::Property<float> m_min_distance_to_beamline {
      this,
      "min_distance_to_beamline",
      100.0f,
      "The cut parameter on the distance to beamline. The value is computed as: max(min_distance, slope * z + offset)"};
    Allen::Property<float> m_min_distance_to_beamline_slope {
      this,
      "min_distance_to_beamline_slope",
      0.01f,
      "The cut parameter on the distance to beamline. The value is computed as: max(min_distance, slope * z + offset)"};
    Allen::Property<float> m_min_p2 {
      this,
      "min_p2",
      10000.f * 10000.f,
      "The minimum momentum of composite particle (MeV)"};
    Allen::Property<float> m_min_pt2 {
      this,
      "min_pt2",
      600.f * 600.f,
      "The minimum transverse momentum of composite particle (MeV)"};

    Allen::Monitoring::Histogram2D<> m_xz_vrtx_pos {
      this,
      "xz_vertex_position",
      "xz_vertex_position",
      {100u, 1000.f, 10000.f},
      {100u, -2000.f, 2000.f}};
    Allen::Monitoring::Histogram2D<> m_yz_vrtx_pos {
      this,
      "yz_vertex_position",
      "yz_vertex_position",
      {100u, 1000.f, 10000.f},
      {100u, -2000.f, 2000.f}};
  };

  __global__ void make_ttrack_vertices(Parameters, make_ttrack_vertices_t::DeviceProperties);
} // namespace make_ttrack_vertices
