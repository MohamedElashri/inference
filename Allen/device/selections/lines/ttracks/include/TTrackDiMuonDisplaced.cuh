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

namespace ttrack_dimuon_displaced {
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

  struct ttrack_dimuon_displaced_t : public SelectionAlgorithm,
                                     Parameters,
                                     CompositeParticleLineWithIndex<ttrack_dimuon_displaced_t, Parameters> {

    struct DeviceProperties {
      float min_p;
      float min_pt;
      float min_track_p;
      float min_max_track_p;
      float max_eta;
      float min_r2;
      float min_m;
      float max_m;
      float max_doca;
      float max_ovtx_z;
      float max_ip2;
      float min_dira;
      bool opposite_sign;

      Allen::Monitoring::Histogram<>::DeviceType histogram_mass;
      Allen::Monitoring::Histogram<>::DeviceType histogram_p;
      Allen::Monitoring::Histogram<>::DeviceType histogram_pt;
      Allen::Monitoring::Histogram<>::DeviceType histogram_eta;

      DeviceProperties(const ttrack_dimuon_displaced_t& algo, const Allen::Context& ctx) :
        min_p(algo.m_min_p.value()), min_pt(algo.m_min_pt.value()), min_track_p(algo.m_min_track_p.value()),
        min_max_track_p(algo.m_min_max_track_p.value()), max_eta(algo.m_max_eta.value()), min_r2(algo.m_min_r2.value()),
        min_m(algo.m_min_m.value()), max_m(algo.m_max_m.value()), max_doca(algo.m_max_doca.value()),
        max_ovtx_z(algo.m_max_ovtx_z.value()), max_ip2(algo.m_max_ip2.value()), min_dira(algo.m_min_dira.value()),
        opposite_sign(algo.m_opposite_sign.value()), histogram_mass(algo.m_histogram_mass.data(ctx)),
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
    Allen::Property<float> m_min_p {this, "min_p", 0.f, "minimum momentum (MeV/c)"};
    Allen::Property<float> m_min_pt {this, "min_pt", 0.f, "minimum transverse momentum (MeV/c)"};
    Allen::Property<float> m_min_track_p {this, "min_track_p", 0.f, "minimum track momentum (MeV/c)"};
    Allen::Property<float> m_min_max_track_p {this,
                                              "min_max_track_p",
                                              0.f,
                                              "minimum momentum of a track that has larger momentum (MeV/c)"};

    Allen::Property<float> m_max_eta {this, "max_eta", 10.f, "maximum eta"}; // Effectively no upper-limit by default
    Allen::Property<float> m_min_r2 {this, "min_r2", 0.f, "minimum distance to the beamline squared (mm^2)"};

    Allen::Property<float> m_min_m {this, "min_m", 0.f, "minimum mass (MeV/c^2)"};
    Allen::Property<float> m_max_m {this,
                                    "max_m",
                                    std::numeric_limits<float>::max(),
                                    "maximum mass (MeV/c^2)"}; // Effectively no upper-limit by default
    Allen::Property<float> m_max_doca {this,
                                       "max_doca",
                                       std::numeric_limits<float>::max(),
                                       "maximum distance of closest approach (mm)"};

    Allen::Property<float> m_max_ovtx_z {this, "max_ovtx_z", 8500.f, "maximum vertex z (mm)"};
    Allen::Property<float> m_max_ip2 {this,
                                      "max_ip2",
                                      std::numeric_limits<float>::max(),
                                      "maximum impact parameter squared of parent particle (mm^2)"};
    Allen::Property<float> m_min_dira {this, "min_dira", 0.f, "minimum cosine of direction angle"};

    Allen::Property<bool> m_opposite_sign {this, "opposite_sign", true, "require opposite sign muons"};

    Allen::Monitoring::Histogram<> m_histogram_mass {this, "mass", "m", {100u, 0.0f, 5000.0f}};
    Allen::Monitoring::Histogram<> m_histogram_p {this, "p", "p", {100u, 0.0f, 50e3f}};
    Allen::Monitoring::Histogram<> m_histogram_pt {this, "pt", "pT", {100u, 0.0f, 10e3f}};
    Allen::Monitoring::Histogram<> m_histogram_eta {this, "eta", "eta", {100u, 1.f, 6.f}};
  };
} // namespace ttrack_dimuon_displaced
