/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
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
#include "CompositeParticleLine.cuh"
#include "MassDefinitions.h"

#include "AllenMonitoring.h"

namespace ks_ll_detached_track_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_svs_t, unsigned) host_number_of_svs;
    DEVICE_INPUT(dev_particle_container_t, Allen::Views::Physics::MultiEventCompositeParticles) dev_particle_container;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    HOST_OUTPUT(host_line_data_t, LineData) host_line_data;
    HOST_OUTPUT_WITH_DEPENDENCIES(host_fn_parameters_t, DEPENDENCIES(dev_particle_container_t), char)
    host_fn_parameters;

    DEVICE_OUTPUT(M_t, float) M;
    DEVICE_OUTPUT(MCORR_t, float) MCORR;
    DEVICE_OUTPUT(PT_t, float) PT;
    DEVICE_OUTPUT(DOCA_t, float) DOCA;
    DEVICE_OUTPUT(VZ_t, float) VZ;
    DEVICE_OUTPUT(BPVVDZ_t, float) BPVVDZ;
    DEVICE_OUTPUT(BPVVDRHO_t, float) BPVVDRHO;
    DEVICE_OUTPUT(BPVDIRA_t, float) BPVDIRA;
    DEVICE_OUTPUT(BPVIP_t, float) BPVIP;
    DEVICE_OUTPUT(BPVFD_t, float) BPVFD;
    DEVICE_OUTPUT(h_P_t, float) h_P;
    DEVICE_OUTPUT(h_PT_t, float) h_PT;
    DEVICE_OUTPUT(h_MIPCHI2_t, float) h_MIPCHI2;
    DEVICE_OUTPUT(h_MIP_t, float) h_MIP;
    DEVICE_OUTPUT(h_CHI2NDF_t, float) h_CHI2NDF;
    DEVICE_OUTPUT(h_Q_t, float) h_Q;
    DEVICE_OUTPUT(pi1_P_t, float) pi1_P;
    DEVICE_OUTPUT(pi1_PT_t, float) pi1_PT;
    DEVICE_OUTPUT(pi1_MIPCHI2_t, float) pi1_MIPCHI2;
    DEVICE_OUTPUT(pi1_MIP_t, float) pi1_MIP;
    DEVICE_OUTPUT(pi1_CHI2NDF_t, float) pi1_CHI2NDF;
    DEVICE_OUTPUT(pi1_Q_t, float) pi1_Q;
    DEVICE_OUTPUT(pi2_P_t, float) pi2_P;
    DEVICE_OUTPUT(pi2_PT_t, float) pi2_PT;
    DEVICE_OUTPUT(pi2_MIPCHI2_t, float) pi2_MIPCHI2;
    DEVICE_OUTPUT(pi2_MIP_t, float) pi2_MIP;
    DEVICE_OUTPUT(pi2_CHI2NDF_t, float) pi2_CHI2NDF;
    DEVICE_OUTPUT(pi2_Q_t, float) pi2_Q;
    DEVICE_OUTPUT(pi1_pi2_DOCA_t, float) pi1_pi2_DOCA;
    DEVICE_OUTPUT(Ks_M_t, float) Ks_M;
    DEVICE_OUTPUT(Ks_VCHI2_t, float) Ks_VCHI2;
    DEVICE_OUTPUT(Ks_VZ_t, float) Ks_VZ;
    DEVICE_OUTPUT(Ks_PT_t, float) Ks_PT;
    DEVICE_OUTPUT(Ks_BPVVDCHI2_t, float) Ks_BPVVDCHI2;
    DEVICE_OUTPUT(Ks_BPVVDZ_t, float) Ks_BPVVDZ;
    DEVICE_OUTPUT(Ks_BPVVDRHO_t, float) Ks_BPVVDRHO;
    DEVICE_OUTPUT(Ks_BPVDIRA_t, float) Ks_BPVDIRA;
    DEVICE_OUTPUT(evtNo_t, uint64_t) evtNo;
    DEVICE_OUTPUT(runNo_t, unsigned) runNo;
  };

  struct ks_ll_detached_track_line_t : public SelectionAlgorithm,
                                       Parameters,
                                       CompositeParticleLine<ks_ll_detached_track_line_t, Parameters> {
    struct DeviceProperties {
      float pi_PT_min;
      float pi_MIPCHI2_min;
      float Ks_PT_min;
      float Ks_DOCA_max;
      float Ks_VCHI2_max;
      float Ks_BPVFDCHI2_min;
      float Ks_M_min;
      float Ks_M_max;
      float h_MIPCHI2_min;
      float h_PT_min;
      float VZ_min;
      float VZ_max;
      float M_max;
      float M_min;
      float DIRA_min;
      float BPVVDZ_min;
      float BPVVDRHO_min;
      float KsVDZ_min;
      float SUMPT_min;
      Allen::Monitoring::Histogram<>::DeviceType histogram_Ks_mass;
      Allen::Monitoring::Histogram<>::DeviceType histogram_KsPi_mass;
      DeviceProperties(const ks_ll_detached_track_line_t& algo, const Allen::Context& ctx) :
        pi_PT_min(algo.m_pi_PT_min), pi_MIPCHI2_min(algo.m_pi_MIPCHI2_min), Ks_PT_min(algo.m_Ks_PT_min),
        Ks_DOCA_max(algo.m_Ks_DOCA_max), Ks_VCHI2_max(algo.m_Ks_VCHI2_max), Ks_BPVFDCHI2_min(algo.m_Ks_BPVFDCHI2_min),
        Ks_M_min(algo.m_Ks_M_min), Ks_M_max(algo.m_Ks_M_max), h_MIPCHI2_min(algo.m_h_MIPCHI2_min),
        h_PT_min(algo.m_h_PT_min), VZ_min(algo.m_VZ_min), VZ_max(algo.m_VZ_max), M_max(algo.m_M_max),
        M_min(algo.m_M_min), DIRA_min(algo.m_DIRA_min), BPVVDZ_min(algo.m_BPVVDZ_min),
        BPVVDRHO_min(algo.m_BPVVDRHO_min), KsVDZ_min(algo.m_KsVDZ_min), SUMPT_min(algo.m_SUMPT_min),
        histogram_Ks_mass(algo.m_histogram_Ks_mass.data(ctx)), histogram_KsPi_mass(algo.m_histogram_KsPi_mass.data(ctx))
      {}
    };

    __device__ static bool
    select(const Parameters&, const DeviceProperties&, std::tuple<const Allen::Views::Physics::CompositeParticle>);

    __device__ static bool fill_tuples(
      const Parameters&,
      const DeviceProperties&,
      std::tuple<const Allen::Views::Physics::CompositeParticle>,
      unsigned,
      bool);
    __device__ static void monitor(
      const Parameters&,
      const DeviceProperties&,
      std::tuple<const Allen::Views::Physics::CompositeParticle>,
      unsigned,
      bool);

    using monitoring_types = std::tuple<
      M_t,
      MCORR_t,
      PT_t,
      DOCA_t,
      VZ_t,
      BPVVDZ_t,
      BPVVDRHO_t,
      BPVDIRA_t,
      BPVIP_t,
      BPVFD_t,
      h_P_t,
      h_PT_t,
      h_MIPCHI2_t,
      h_MIP_t,
      h_CHI2NDF_t,
      h_Q_t,
      pi1_P_t,
      pi1_PT_t,
      pi1_MIPCHI2_t,
      pi1_MIP_t,
      pi1_CHI2NDF_t,
      pi1_Q_t,
      pi2_P_t,
      pi2_PT_t,
      pi2_MIPCHI2_t,
      pi2_MIP_t,
      pi2_CHI2NDF_t,
      pi2_Q_t,
      pi1_pi2_DOCA_t,
      Ks_M_t,
      Ks_VCHI2_t,
      Ks_VZ_t,
      Ks_PT_t,
      Ks_BPVVDCHI2_t,
      Ks_BPVVDZ_t,
      Ks_BPVVDRHO_t,
      Ks_BPVDIRA_t,
      evtNo_t,
      runNo_t>;

  private:
    Allen::Property<float> m_pi_PT_min {
      this,
      "pi_PT_min",
      470.f * Allen::Units::MeV,
      "min pT of each pion candidate"}; // modified wrt LambdaLLDetachedTrackLine(Allen/v7r6p2)
    Allen::Property<float> m_pi_MIPCHI2_min {this, "pi_MIPCHI2_min", 64.f, "min IP chi2 of each pion candidate"};
    Allen::Property<float> m_Ks_PT_min {this, "Ks_PT_min", 1400.f * Allen::Units::MeV, "pT of Ks candidate"};
    Allen::Property<float> m_Ks_DOCA_max {this, "Ks_DOCA_max", 100.f * Allen::Units::um, "DOCA of Ks daughters"};
    Allen::Property<float> m_Ks_VCHI2_max {this, "Ks_VCHI2_max", 16.f, "Ks vertex chi2"};
    Allen::Property<float> m_Ks_BPVFDCHI2_min {this, "Ks_BPVFDCHI2_min", 240.f, "flight distance chi2 of Ks candidate"};
    Allen::Property<float> m_Ks_M_min {
      this,
      "Ks_M_min",
      455.f * Allen::Units::MeV,
      "min mass of Ks candidate"}; // modified wrt LambdaLLDetachedTrackLine(Allen/v7r6p2)
    Allen::Property<float> m_Ks_M_max {
      this,
      "Ks_M_max",
      545.f * Allen::Units::MeV,
      "max mass of Ks candidate"}; // modified wrt LambdaLLDetachedTrackLine(Allen/v7r6p2)
    Allen::Property<float> m_h_MIPCHI2_min {this, "h_MIPCHI2_min", 12.f, "min companion track IP chi2"};
    Allen::Property<float> m_h_PT_min {this, "h_PT_min", 700.f * Allen::Units::MeV, "min companion track PT"};
    Allen::Property<float> m_VZ_min {this, "VZ_min", -200.f * Allen::Units::mm, "min vertex z position"};
    Allen::Property<float> m_VZ_max {this, "VZ_max", 250.f * Allen::Units::mm, "max vertex z position"};
    Allen::Property<float> m_M_max {
      this,
      "M_max",
      2.1f * Allen::Units::GeV,
      "max mass given pion mass hypothesis for companion track"}; // modified wrt
                                                                  // LambdaLLDetachedTrackLine(Allen/v7r6p2)
    Allen::Property<float> m_M_min {
      this,
      "M_min",
      1.5f * Allen::Units::GeV,
      "min mass given pion mass hypothesis for companion track"}; // new wrt LambdaLLDetachedTrackLine(Allen/v7r6p2)
    Allen::Property<float> m_DIRA_min {
      this,
      "DIRA_min",
      0.999,
      "min cos(DIRA) for Ks +  track composite"}; // new wrt LambdaLLDetachedTrackLine(Allen/v7r6p2)
    Allen::Property<float> m_BPVVDZ_min {this, "BPVVDZ_min", 1.5f * Allen::Units::mm, "min distance (in z) to best PV"};
    Allen::Property<float> m_BPVVDRHO_min {this, "BPVVDRHO_min", 120.f * Allen::Units::um, "min distance to best PV"};
    Allen::Property<float> m_KsVDZ_min {this, "KsVDZ_min", 8.f * Allen::Units::mm, "min distance (in z) to Ks vertex"};
    Allen::Property<float> m_SUMPT_min {
      this,
      "SUMPT_min",
      2500.f * Allen::Units::MeV,
      "sum of Ks and track pT"}; // modified wrt LambdaLLDetachedTrackLine(Allen/v7r6p2)

    Allen::Monitoring::Histogram<> m_histogram_Ks_mass {
      this,
      "Ks_mass_KsLLDetachedTrack",
      "m(#pi^{+}#pi^{#minus}) [MeV]",
      {100u, 400.f * Allen::Units::MeV, 600.f * Allen::Units::MeV}};
    Allen::Monitoring::Histogram<> m_histogram_KsPi_mass {
      this,
      "KsTrack_mass_KsLLDetachedTrack",
      "m(Ks#pi^{+}) [MeV]",
      {94u, 1.25f * Allen::Units::GeV, 2.6f * Allen::Units::GeV}};
  };
} // namespace ks_ll_detached_track_line
