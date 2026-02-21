/*****************************************************************************\
* (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#pragma once

#include "AlgorithmTypes.cuh"
#include "CompositeParticleLineWithIndex.cuh"
#include "ROOTService.h"
#include "MassDefinitions.h"
#include "AllenMonitoring.h"

namespace ttrack_ks2pipi {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_svs_t, unsigned) host_number_of_svs;
    DEVICE_INPUT(dev_particle_container_t, Allen::Views::Physics::MultiEventCompositeParticles) dev_particle_container;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    HOST_OUTPUT(host_line_data_t, LineData) host_line_data;
    HOST_OUTPUT_WITH_DEPENDENCIES(host_fn_parameters_t, DEPENDENCIES(dev_particle_container_t), char)
    host_fn_parameters;

    DEVICE_OUTPUT(mon_mass_t, float) mon_mass;
    DEVICE_OUTPUT(mon_p_t, float) mon_p;
    DEVICE_OUTPUT(mon_pt_t, float) mon_pt;
    DEVICE_OUTPUT(mon_eta_t, float) mon_eta;
    DEVICE_OUTPUT(mon_vrtx_x_t, float) mon_vrtx_x;
    DEVICE_OUTPUT(mon_vrtx_y_t, float) mon_vrtx_y;
    DEVICE_OUTPUT(mon_vrtx_z_t, float) mon_vrtx_z;
    DEVICE_OUTPUT(evtNo_t, uint64_t) evtNo;
    DEVICE_OUTPUT(runNo_t, unsigned) runNo;
  };

  struct ttrack_ks2pipi_t : public SelectionAlgorithm,
                            Parameters,
                            CompositeParticleLineWithIndex<ttrack_ks2pipi_t, Parameters> {

    struct DeviceProperties {
      float max_doca;
      float min_p;
      float min_pt;
      float max_m;
      float max_ip2;
      float min_r2;
      float min_track_p;
      float max_ovtx_z;
      float min_dira;

      Allen::Monitoring::Histogram<>::DeviceType histogram_mass;
      Allen::Monitoring::Histogram<>::DeviceType histogram_p;
      Allen::Monitoring::Histogram<>::DeviceType histogram_pt;
      Allen::Monitoring::Histogram<>::DeviceType histogram_eta;

      DeviceProperties(const ttrack_ks2pipi_t& algo, const Allen::Context& ctx) :
        max_doca(algo.m_max_doca.value()), min_p(algo.m_min_p.value()), min_pt(algo.m_min_pt.value()),
        max_m(algo.m_max_m.value()), max_ip2(algo.m_max_ip2.value()), min_r2(algo.m_min_r2.value()),
        min_track_p(algo.m_min_track_p.value()), max_ovtx_z(algo.m_max_ovtx_z.value()),
        min_dira(algo.m_min_dira.value()), histogram_mass(algo.m_histogram_mass.data(ctx)),
        histogram_p(algo.m_histogram_p.data(ctx)), histogram_pt(algo.m_histogram_pt.data(ctx)),
        histogram_eta(algo.m_histogram_eta.data(ctx))
      {}
    };

    using monitoring_types =
      std::tuple<mon_mass_t, mon_p_t, mon_pt_t, mon_eta_t, mon_vrtx_x_t, mon_vrtx_y_t, mon_vrtx_z_t, evtNo_t, runNo_t>;

    __device__ static bool select(
      const Parameters&,
      const DeviceProperties&,
      std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned>);

    __device__ static void monitor(
      const Parameters&,
      const DeviceProperties&,
      std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned>,
      unsigned,
      bool);

    __device__ static bool fill_tuples(
      const Parameters&,
      const DeviceProperties&,
      std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned> input,
      unsigned index,
      bool sel);

  private:
    Allen::Property<float> m_max_doca {this, "max_doca", 20.f, "maximum distance of closest approach"};
    Allen::Property<float> m_max_m {this, "max_m", 1200.f, "maximum mass"};
    Allen::Property<float> m_min_p {this, "min_p", 62534.2f, "minimum momentum"};
    Allen::Property<float> m_min_pt {this, "min_pt", 4072.2f, "minimum transverse momentum"};
    Allen::Property<float> m_min_r2 {this, "min_r2", 257.2f * 257.2f, "minimum distance to the beamline"};
    Allen::Property<float> m_min_track_p {this, "min_track_p", 32795.9f, "minimum track momentum"};
    Allen::Property<float> m_max_ip2 {this, "max_ip2", 80.f * 80.f, "maximum impact parameter squared"};
    Allen::Property<float> m_min_dira {this, "min_dira", 0.999f, "minimum direction angle cosine"};
    Allen::Property<float> m_max_ovtx_z {this, "max_ovtx_z", 8500.f, "maximum z of the vertex"};

    Allen::Monitoring::Histogram<> m_histogram_mass {this, "mass", "m", {100u, 0.0f, 1200.0f}};
    Allen::Monitoring::Histogram<> m_histogram_p {this, "p", "p", {100u, 0.0f, 50e3f}};
    Allen::Monitoring::Histogram<> m_histogram_pt {this, "pt", "pT", {100u, 0.0f, 10e3f}};
    Allen::Monitoring::Histogram<> m_histogram_eta {this, "eta", "eta", {100u, 1.f, 6.f}};
  };
} // namespace ttrack_ks2pipi
