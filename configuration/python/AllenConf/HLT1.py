###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################

from PyConf.tonic import configurable

from AllenConf.enum_types import TrackingType
from AllenConf.filters import *
from AllenConf.get_thresholds import get_thresholds
from AllenConf.hlt1_calibration_lines import *
from AllenConf.hlt1_charged_kaon_lines import *
from AllenConf.hlt1_charm_lines import *
from AllenConf.HLT1_common import *
from AllenConf.hlt1_downstream_lines import *
from AllenConf.hlt1_electron_lines import *
from AllenConf.hlt1_inclusive_hadron_lines import *
from AllenConf.hlt1_muon_lines import *
from AllenConf.hlt1_photon_lines import make_diphotonhighmass_line
from AllenConf.hlt1_reconstruction import hlt1_reconstruction
from AllenConf.hlt1_smog2_lines import *
from AllenConf.hlt1_ttrack_lines import *
from AllenConf.odin import make_bxtype, make_event_type, make_nzs_filter
from AllenConf.utils import line_maker


def default_physics_lines(
    reconstructed_objects,
    with_calo,
    with_muon,
    with_v0s,
    with_quirks,
    thresholds,
    enable_tupling,
    chi2_cuts,
):
    velo_tracks = reconstructed_objects["velo_tracks"]
    long_tracks = reconstructed_objects["long_tracks"]
    long_track_particles = reconstructed_objects["long_track_particles"]
    pvs = reconstructed_objects["pvs"]
    dihadrons = reconstructed_objects["dihadron_secondary_vertices"]
    prompt_dihadrons = reconstructed_objects["prompt_dihadron_secondary_vertices"]
    dileptons = reconstructed_objects["dilepton_secondary_vertices"]
    dileptons_nopt = reconstructed_objects["dilepton_secondary_vertices_nopt"]
    v0s = reconstructed_objects["v0_secondary_vertices"]
    lambda_track_from_c = reconstructed_objects["lambda_track_from_c"]
    ks_track_from_c = reconstructed_objects["ks_track_from_c"]
    v0_twotrack_pairs = reconstructed_objects["v0_sv_twotrack_pairs"]
    dstars = reconstructed_objects["dstars"]
    three_body_svs = reconstructed_objects["three_body_svs"]
    v0_pairs = reconstructed_objects["v0_pairs"]
    v0_hh_pairs = reconstructed_objects["v0_hh_pairs"]
    phi_plus_track = reconstructed_objects["phi_plus_track"]
    dihadrons_noipcut = reconstructed_objects["dihadrons_noipcut"]  # noqa: F841
    muon_stubs = reconstructed_objects["muon_stubs"]  # noqa: F841

    lines = [
        make_track_mva_line(
            long_tracks,
            long_track_particles,
            maxChi2Ndof=chi2_cuts.Hlt1TrackMVA_maxChi2Ndof,
            name="Hlt1TrackMVA",
            enable_tupling=enable_tupling,
            alpha=thresholds.TrackMVA_alpha,
            maxGhostProb=thresholds.TrackMVA_maxGhostProb,
        ),
        make_two_track_mva_line(
            long_tracks,
            dihadrons,
            name="Hlt1TwoTrackMVA",
            enable_tupling=enable_tupling,
            minMVA=thresholds.TwoTrackMVA_minMVA,
            maxGhostProb=thresholds.TwoTrackMVA_maxGhostProb,
        ),
        make_d2kk_line(
            long_tracks,
            dihadrons,
            name="Hlt1D2KK",
            enable_tupling=enable_tupling,
            charm_track_ip=thresholds.D2HH_track_ip,
            charm_track_pt=thresholds.D2HH_track_pt,
        ),
        make_d2kpi_line(
            long_tracks,
            dihadrons,
            name="Hlt1D2KPi",
            enable_tupling=enable_tupling,
            charm_track_ip=thresholds.D2HH_track_ip,
            charm_track_pt=thresholds.D2HH_track_pt,
        ),
        make_d2pipi_line(
            long_tracks,
            dihadrons,
            name="Hlt1D2PiPi",
            enable_tupling=enable_tupling,
            charm_track_ip=thresholds.D2HH_track_ip,
            charm_track_pt=thresholds.D2HH_track_pt,
        ),
        make_dst_line(dstars, name="Hlt1Dst2D0Pi", enable_tupling=enable_tupling),
        make_diproton_highmass_line(
            prompt_dihadrons,
            name="Hlt1DiProtonHighMass",
            minPT_p=thresholds.DiProtonHighMass_P_minPt,
            minPT_pp=thresholds.DiProtonHighMass_PP_minPt,
            pre_scaler=1.0,
            enable_tupling=enable_tupling,
        ),
        make_kplus_to_piee_line(
            three_body_svs,
            name="Hlt1Kplus2PiEE",
            enable_monitoring=False,
            enable_tupling=enable_tupling,
        ),
        make_kplus_to_pimumu_line(
            three_body_svs,
            name="Hlt1Kplus2PiMuMu",
            enable_monitoring=False,
            enable_tupling=enable_tupling,
        ),
        make_kplus_to_3pi_line(
            three_body_svs,
            name="Hlt1Kplus2PiPiPi",
            enable_monitoring=False,
            enable_tupling=enable_tupling,
        ),
        make_tautophimu_line(
            phi_plus_track, name="Hlt1TauToPhiMu", enable_tupling=enable_tupling
        ),
    ]
    if with_quirks:
        lines += [
            make_quirks_line(
                maxPHI=thresholds.Quirks_maxPHI,
                maxPHIDF=thresholds.Quirks_maxPHIDF,
                maxR=thresholds.Quirks_maxR,
                minStations=thresholds.Quirks_minStations,
                hit_threshold=thresholds.Quirks_hit_thresholds,
                max_opposite_considered=thresholds.Quirks_max_opposite_considered,
                name="Hlt1Quirks",
                enable_tupling=enable_tupling,
            ),
        ]

    if (
        "downstream_tracks" in reconstructed_objects
        and "downstream_secondary_vertices" in reconstructed_objects
    ):
        lines += [
            make_downstream_kshort_line(
                reconstructed_objects["downstream_tracks"],
                reconstructed_objects["downstream_secondary_vertices"],
                mva_ks_threshold=0.55,
                mva_detached_ks_threshold=thresholds.DownstreamKsToPiPi_minMVA_detached,
                name="Hlt1DownstreamKsToPiPi",
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_downstream_lambda_line(
                reconstructed_objects["downstream_tracks"],
                reconstructed_objects["downstream_secondary_vertices"],
                mva_l0_threshold=0.5,
                mva_detached_l0_threshold=thresholds.DownstreamLambdaToPPi_minMVA_detached,
                name="Hlt1DownstreamLambdaToPPi",
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_downstream_kshort_line(
                reconstructed_objects["downstream_tracks"],
                reconstructed_objects["downstream_secondary_vertices"],
                post_scaler=0.001,
                mva_ks_threshold=0.5,
                mva_detached_ks_threshold=0.0,
                name="Hlt1DownstreamPromptKsToPiPi",
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_downstream_lambda_line(
                reconstructed_objects["downstream_tracks"],
                reconstructed_objects["downstream_secondary_vertices"],
                post_scaler=0.001,
                mva_l0_threshold=0.5,
                mva_detached_l0_threshold=0.0,
                name="Hlt1DownstreamPromptLambdaToPPi",
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_downstream_gamma_line(
                reconstructed_objects["downstream_tracks"],
                reconstructed_objects["downstream_secondary_vertices"],
                minPt=thresholds.DownstreamGammaToEE_minPt,
                name="Hlt1DownstreamGammaToEE",
                post_scaler=1.0,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 Monitoring Line / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaMonitoring",
                line_type="monitoring",
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 Monitoring Line with same sign reconstuction / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_same_sign_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaMonitoringSameSign",
                line_type="monitoring",
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 Monitoring Line for clean region / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaMonitoringCleanRegion",
                line_type="monitoring",
                histogram_ks_mass_min=800,
                histogram_ks_mass_nbins=50,
                histogram_ks_fd_min=1200,
                histogram_ks_fd_max=1600,
                histogram_ks_fd_nbins=10,
                clean_region=True,
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 Monitoring Line for clean region with same sign reconstuction / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_same_sign_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaMonitoringCleanRegionSameSign",
                line_type="monitoring",
                enable_trigger=False,
                histogram_ks_mass_min=800,
                histogram_ks_mass_nbins=50,
                histogram_ks_fd_min=1200,
                histogram_ks_fd_max=1600,
                histogram_ks_fd_nbins=10,
                clean_region=True,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 HH with PiPi mass hypo. / Trigger enabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaPostScaled",
                line_type="hadron",
                enable_trigger=True,
                post_scaler=0.005,
                trigger_mass_max=100_000,  # 100 GeV maximum mass
                mva_busca_threshold=0.1,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 MuMu Line / Trigger enabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaMuMuLine",
                line_type="muon",
                mva_busca_threshold=0.1,
                trigger_mass_max=100_000,
                enable_trigger=True,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 ElEl Line / Trigger enabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaElElLine",
                line_type="electron",
                mva_busca_threshold=0.1,
                trigger_mass_max=100_000,
                enable_trigger=True,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 KK Line / Trigger enabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaKKlLine",
                line_type="kaon",
                mva_busca_threshold=0.97,
                enable_trigger=True,
                trigger_mass_min=1200,
                trigger_mass_max=5000,
                trigger_fd_min=1200,
                trigger_fd_max=2000,
                disable_R_cut=True,
                enable_tupling=enable_tupling,
            ),
            make_downstream_two_track_ks_line(
                reconstructed_objects["downstream_tracks"],
                reconstructed_objects["downstream_secondary_vertices"],
                name="Hlt1DownstreamTwoTrackKs",
                minTrackPt_piKs=thresholds.DownstreamTwoTrackKs_minTrackPt_piKs,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 HH with PiPi mass hypo. with same sign reconstuction / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_same_sign_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaPostScaledSameSign",
                line_type="hadron",
                post_scaler=0.01,
                mva_busca_threshold=0.1,
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 MuMu Line with same sign reconstuction / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_same_sign_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaMuMuLineSameSign",
                line_type="muon",
                mva_busca_threshold=0.1,
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 ElEl Line with same sign reconstuction / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_same_sign_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaElElLineSameSign",
                line_type="electron",
                mva_busca_threshold=0.1,
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 High Mass Line / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaHighMassLine",
                line_type="monitoring",
                histogram_ks_fd_min=500,
                histogram_ks_fd_max=2500,
                histogram_ks_fd_nbins=20,
                histogram_ks_mass_min=1200,
                histogram_ks_mass_max=50000,
                histogram_ks_mass_nbins=50,
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 High Mass Line with same sign reconstuction / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_same_sign_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaHighMassLineSameSign",
                line_type="monitoring",
                histogram_ks_fd_min=500,
                histogram_ks_fd_max=2500,
                histogram_ks_fd_nbins=20,
                histogram_ks_mass_min=1200,
                histogram_ks_mass_max=50000,
                histogram_ks_mass_nbins=50,
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 MuMu High Mass Line / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaMuMuHighMassLine",
                line_type="muon",
                histogram_ks_fd_min=500,
                histogram_ks_fd_max=2500,
                histogram_ks_fd_nbins=20,
                histogram_ks_mass_min=1200,
                histogram_ks_mass_max=50000,
                histogram_ks_mass_nbins=50,
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
            make_BuSca_line(  # BuSca HLT1 MuMu High Mass Line with same sign reconstuction / Trigger disabled
                reconstructed_objects[
                    "downstream_combined_hadronic_and_leptonic_same_sign_secondary_vertices"
                ],
                name="Hlt1DownstreamBuScaMuMuHighMassLineSameSign",
                line_type="muon",
                histogram_ks_fd_min=500,
                histogram_ks_fd_max=2500,
                histogram_ks_fd_nbins=20,
                histogram_ks_mass_min=1200,
                histogram_ks_mass_max=50000,
                histogram_ks_mass_nbins=50,
                enable_trigger=False,
                enable_tupling=enable_tupling,
            ),
        ]
        if "downstream_sv_pairs" in reconstructed_objects:
            lines += [
                make_d02ksks_DDDD_line(
                    reconstructed_objects["downstream_tracks"],
                    reconstructed_objects["downstream_sv_pairs"],
                    name="Hlt1D02KsKsDDDD",
                    minTrackPt_piKs=450.0,
                    minComboPt_Ks=1200.0,
                    enable_tupling=enable_tupling,
                ),
            ]

    if "ttrack_vertices" in reconstructed_objects:
        lines += [
            make_ttrack_highmass_dimuon_displaced_line(
                reconstructed_objects["ttrack_vertices"],
                name="Hlt1TTrackHighMassDiMuonDisplaced",
                enable_tupling=enable_tupling,
            ),
            make_ttrack_lowmass_dimuon_displaced_line(
                reconstructed_objects["ttrack_vertices"],
                name="Hlt1TTrackLowMassDiMuonDisplaced",
                enable_tupling=enable_tupling,
            ),
            make_ttrack_highmass_dimuon_displaced_samesign_line(
                reconstructed_objects["ttrack_vertices"],
                name="Hlt1TTrackHighMassDiMuonDisplacedSameSign",
                enable_tupling=enable_tupling,
            ),
            make_ttrack_lowmass_dimuon_displaced_samesign_line(
                reconstructed_objects["ttrack_vertices"],
                name="Hlt1TTrackLowMassDiMuonDisplacedSameSign",
                enable_tupling=enable_tupling,
            ),
            make_ttrack_lambda2ppi_line(
                reconstructed_objects["ttrack_vertices"],
                name="Hlt1TTrackLambda2PPi",
                enable_tupling=enable_tupling,
            ),
            make_ttrack_ks2pipi_line(
                reconstructed_objects["ttrack_vertices"],
                name="Hlt1TTrackKs2PiPi",
                enable_tupling=enable_tupling,
            ),
        ]

    if "v0dd_hh_pairs" in reconstructed_objects:
        lines += [
            make_d2kshh_line(
                long_tracks,
                reconstructed_objects["v0dd_hh_pairs"],
                maxVertexChi2=10,
                pre_scaler=0.25,
                maxDOCA=2.5,
                minM_Ks=420.0,
                maxM_Ks=540.0,
                minTrackIP_Ks=0.2,
                minCTau_D0=0.5 * 0.1229,
                name="Hlt1DownstreamD2Kshh",
                enable_tupling=enable_tupling,
            ),
        ]

    if with_v0s:
        lines += [
            make_kstopipi_line(
                long_tracks,
                v0s,
                name="Hlt1KsToPiPi",
                post_scaler=0.001,
                enable_tupling=enable_tupling,
            ),
            make_kstopipi_line(
                long_tracks,
                v0s,
                name="Hlt1KsToPiPiDoubleMuonMisID",
                double_muon_misid=True,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_two_track_line_ks(
                long_tracks,
                v0s,
                name="Hlt1TwoTrackKs",
                minTrackPt_piKs=thresholds.TwoTrackKs_minTrackPt_piKs,
                minTrackIPChi2_Ks=thresholds.TwoTrackKs_minTrackIPChi2_piKs,
                maxEta_Ks=thresholds.TwoTrackKs_maxEta_Ks,
                min_combip=thresholds.TwoTrackKs_min_combip,
                minComboPt_Ks=thresholds.TwoTrackKs_minComboPt_Ks,
                enable_tupling=enable_tupling,
            ),
            make_two_ks_line(
                long_tracks, v0_pairs, name="Hlt1TwoKs", enable_tupling=enable_tupling
            ),
            make_lambda_ll_detached_track_line(
                lambda_track_from_c,
                name="Hlt1LambdaLLDetachedTrack",
                enable_tupling=enable_tupling,
            ),
            make_ks_ll_detached_track_line(
                ks_track_from_c,
                name="Hlt1KsLLDetachedTrack",
                enable_tupling=enable_tupling,
            ),
            make_detached_xi_omega_lll_line(
                v0_twotrack_pairs, name="Hlt1XiOmegaLLL", enable_tupling=enable_tupling
            ),
            make_d2kshh_line(
                long_tracks,
                v0_hh_pairs,
                name="Hlt1D2Kshh",
                enable_tupling=enable_tupling,
                minCTau_D0=0.5 * 0.1229,
            ),
        ]

    if with_muon:
        muonid = reconstructed_objects["muonID"]
        muon_nn_cuts = {
            "vloose": 0.01,  # equivalent to Chi2Corr 2.4 provided PT cuts are loosened too
            "loose": 0.03,  # equivalent to Chi2Corr 1.8 provided PT cuts are loosened too
        }
        lines += [
            make_single_high_pt_muon_line(
                long_tracks,
                long_track_particles,
                maxChi2Ndof=chi2_cuts.Hlt1SingleHighPtMuon_maxChi2Ndof,
                name="Hlt1SingleHighPtMuon",
                enable_tupling=enable_tupling,
                singleMinPt=thresholds.SingleHighPtLepton_pt,
            ),
            make_single_high_pt_muon_no_muid_line(
                long_tracks,
                long_track_particles,
                maxChi2Ndof=chi2_cuts.Hlt1SingleHighPtMuonNoMuID_maxChi2Ndof,
                name="Hlt1SingleHighPtMuonNoMuID",
                enable_tupling=enable_tupling,
                singleMinPt=thresholds.SingleHighPtLepton_pt_noMuonID,
            ),
            make_di_muon_mass_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonHighMass",
                enable_tupling=enable_tupling,
                minHighMassTrackPt=thresholds.DiMuonHighMass_pt,
                useMuonNN=True,
                minMuonNN=thresholds.DiMuonHighMass_NN,
            ),
            make_di_muon_mass_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonDisplaced",
                minHighMassTrackPt=thresholds.DiMuonDisplaced_pt,
                minHighMassTrackP=3000.0,
                minMass=0,
                maxDoca=0.2,
                maxVertexChi2=25.0,
                minIPChi2=thresholds.DiMuonDisplaced_ipchi2,
                enable_tupling=enable_tupling,
                minMuonNN=thresholds.DiMuonDisplaced_NN,
                useMuonNN=True,
            ),
            make_di_muon_mass_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonDisplacedSoftPT",
                minHighMassTrackPt=0,
                minHighMassTrackP=3000.0,
                minMass=200,
                maxDoca=0.2,
                maxVertexChi2=9,
                minIPChi2=6,
                enable_tupling=enable_tupling,
                minMuonNN=thresholds.DiMuonDisplacedSoftPT_NN,
                useMuonNN=True,
                vetoSharedHits=True,
            ),
            make_di_muon_mass_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonDisplacedSoftPT_SS",
                minHighMassTrackPt=0,
                minHighMassTrackP=3000.0,
                minMass=200,
                maxDoca=0.2,
                maxVertexChi2=9,
                minIPChi2=6,
                enable_tupling=enable_tupling,
                minMuonNN=thresholds.DiMuonDisplacedSoftPT_NN,
                useMuonNN=True,
                vetoSharedHits=True,
                oppositeSign=False,
                pre_scaler=0.01,
            ),
            make_di_muon_soft_line(
                long_tracks,
                dileptons_nopt,
                name="Hlt1DiMuonSoft",
                enable_tupling=enable_tupling,
            ),
            make_track_muon_mva_line(
                long_tracks,
                long_track_particles,
                muonid,
                maxChi2Ndof=chi2_cuts.Hlt1TrackMuonMVA_maxChi2Ndof,
                maxChi2Corr=1.8,
                useMuonNN=True,
                minMuonNN=thresholds.TrackMuonMVA_NN,
                name="Hlt1TrackMuonMVA",
                enable_tupling=enable_tupling,
                alpha=thresholds.TrackMuonMVA_alpha,
            ),
            make_di_muon_no_ip_line(
                long_tracks,
                dileptons,
                muonid,
                maxVertexChi2=9,
                minNN=thresholds.DiMuonNoIP_NN,
                maxTrChi2=chi2_cuts.Hlt1DiMuonNoIP_maxTrChi2,
                enable_tupling=enable_tupling,
            ),
            make_di_muon_no_ip_line(
                long_tracks,
                dileptons,
                muonid,
                maxVertexChi2=9,
                minNN=thresholds.DiMuonNoIP_NN,
                maxTrChi2=chi2_cuts.Hlt1DiMuonNoIP_maxTrChi2,
                name="Hlt1DiMuonNoIP_SS",
                pre_scaler_hash_string="di_muon_no_ip_ss_line_pre",
                post_scaler_hash_string="di_muon_no_ip_ss_line_post",
                enable_tupling=enable_tupling,
                ss_on=True,
                post_scaler=0.01,
            ),
            make_di_muon_mass_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonNoIPNorm",
                minHighMassTrackPt=0,
                minHighMassTrackP=3000.0,
                minMass=200,
                maxDoca=0.2,
                maxVertexChi2=9,
                minIPChi2=-1,
                enable_tupling=enable_tupling,
                minMuonNN=thresholds.DiMuonDisplacedSoftPT_NN,
                useMuonNN=True,
                vetoSharedHits=True,
                pre_scaler=0.001,
            ),
            make_di_muon_mass_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonNoIPNorm_SS",
                minHighMassTrackPt=0,
                minHighMassTrackP=3000.0,
                minMass=200,
                maxDoca=0.2,
                maxVertexChi2=9,
                minIPChi2=-1,
                enable_tupling=enable_tupling,
                minMuonNN=thresholds.DiMuonDisplacedSoftPT_NN,
                useMuonNN=True,
                vetoSharedHits=True,
                oppositeSign=False,
                pre_scaler=0.001,
            ),
            make_di_muon_drell_yan_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonDrellYan_VLowMass",
                pre_scaler_hash_string="di_muon_drell_yan_vlow_mass_line_pre",
                post_scaler_hash_string="di_muon_drell_yan_vlow_mass_line_post",
                minMass=2900.0,
                maxMass=5000.0,
                maxChi2Corr=1.8,
                minMuonNN=muon_nn_cuts["loose"],
                useMuonNN=False,
                minTrackP=10000,
                minTrackPt=1000,
                pre_scaler=0.4,
                enable_tupling=enable_tupling,
            ),
            make_di_muon_drell_yan_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonDrellYan_VLowMass_SS",
                pre_scaler_hash_string="di_muon_drell_yan_vlow_mass_SS_line_pre",
                post_scaler_hash_string="di_muon_drell_yan_vlow_mass_SS_line_post",
                minMass=2900.0,  # low enough to capture the J/psi
                maxMass=5000.0,
                maxChi2Corr=1.8,
                minMuonNN=muon_nn_cuts["loose"],
                useMuonNN=False,
                minTrackP=10000,
                minTrackPt=1000,
                pre_scaler=0.4,
                OppositeSign=False,
                enable_tupling=enable_tupling,
            ),
            make_di_muon_drell_yan_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonDrellYan",
                pre_scaler_hash_string="di_muon_drell_yan_line_pre",
                post_scaler_hash_string="di_muon_drell_yan_line_post",
                minMass=5000.0,
                minTrackP=12500,
                maxChi2Corr=2.4,
                minMuonNN=muon_nn_cuts["vloose"],
                useMuonNN=False,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_di_muon_drell_yan_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonDrellYan_SS",
                pre_scaler_hash_string="di_muon_drell_yan_SS_line_pre",
                post_scaler_hash_string="di_muon_drell_yan_SS_line_post",
                minMass=5000.0,
                minTrackP=12500,
                maxChi2Corr=2.4,
                minMuonNN=muon_nn_cuts["vloose"],
                useMuonNN=False,
                OppositeSign=False,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_det_jpsitomumu_tap_line(
                long_tracks,
                dihadrons,
                name="Hlt1DetJpsiToMuMuPosTagLine",
                posTag=True,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
            make_det_jpsitomumu_tap_line(
                long_tracks,
                dihadrons,
                name="Hlt1DetJpsiToMuMuNegTagLine",
                posTag=False,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
        ]

    if with_calo:
        ecal_clusters = reconstructed_objects["ecal_clusters"]
        calo_matching_objects = reconstructed_objects["calo_matching_objects"]
        electronid_nn = reconstructed_objects["electronid_nn"]
        jets = reconstructed_objects["jets"]

        lines += [
            make_track_electron_mva_line(
                long_tracks,
                long_track_particles,
                calo_matching_objects,
                electronid_nn,
                maxChi2Ndof=chi2_cuts.Hlt1TrackElectronMVA_maxChi2Ndof,
                name="Hlt1TrackElectronMVA",
                alpha=thresholds.TrackElectronMVA_alpha,
                useNN=True,  # it uses it implicitely per the definition of isElectron
                minElectronNN=thresholds.TrackElectronMVA_NN,
                enable_tupling=enable_tupling,
            ),
            make_single_high_pt_electron_line(
                long_tracks,
                long_track_particles,
                calo_matching_objects,
                maxChi2Ndof=chi2_cuts.Hlt1SingleHighPtElectron_maxChi2Ndof,
                name="Hlt1SingleHighPtElectron",
                singleMinPt=thresholds.SingleHighPtLepton_pt,
                enable_tupling=enable_tupling,
            ),
            make_displaced_dielectron_line(
                long_tracks,
                dileptons,
                calo_matching_objects,
                electronid_nn,
                useNN=True,  # it uses it implicitely per the definition of isElectron
                minElectronNN=thresholds.DiElectronDisplaced_NN,
                name="Hlt1DiElectronDisplaced",
                MinPT=thresholds.DiElectronDisplaced_pt,
                MinIPChi2=thresholds.DiElectronDisplaced_ipchi2,
                enable_tupling=enable_tupling,
            ),
            make_diphotonhighmass_line(
                ecal_clusters,
                velo_tracks,
                pvs,
                name="Hlt1DiPhotonHighMass",
                enable_tupling=enable_tupling,
                minET=thresholds.DiPhotonHighMass_minET,
            ),
            make_pi02gammagamma_middleoutermixed_line(
                ecal_clusters,
                velo_tracks,
                pvs,
                name="Hlt1Pi02GammaGammaMiddleOuterMixed",
                pre_scaler_hash_string="p02gammagamma_middleoutermixed_line_pre",
                post_scaler_hash_string="p02gammagamma_middleoutermixed_line_post",
                enable_tupling=enable_tupling,
            ),
            make_pi02gammagamma_middleinnermixed_line(
                ecal_clusters,
                velo_tracks,
                pvs,
                name="Hlt1Pi02GammaGammaMiddleInnerMixed",
                pre_scaler_hash_string="p02gammagamma_middleInnermixed_line_pre",
                post_scaler_hash_string="p02gammagamma_middleInnermixed_line_post",
                enable_tupling=enable_tupling,
            ),
            make_pi02gammagamma_line(
                ecal_clusters,
                velo_tracks,
                pvs,
                name="Hlt1Pi02GammaGamma",
                pre_scaler_hash_string="p02gammagamma_line_pre",
                post_scaler_hash_string="p02gammagamma_line_post",
                enable_tupling=enable_tupling,
            ),
            make_pi02gammagamma_outer_line(
                ecal_clusters,
                velo_tracks,
                pvs,
                name="Hlt1Pi02GammaGammaOuter",
                pre_scaler_hash_string="p02gammagamma_outer_line_pre",
                post_scaler_hash_string="p02gammagamma_outer_line_post",
                enable_tupling=enable_tupling,
            ),
            make_pi02gammagamma_middle_line(
                ecal_clusters,
                velo_tracks,
                pvs,
                name="Hlt1Pi02GammaGammaMiddle",
                pre_scaler_hash_string="p02gammagamma_middle_line_pre",
                post_scaler_hash_string="p02gammagamma_middle_line_post",
                enable_tupling=enable_tupling,
            ),
            make_pi02gammagamma_inner_line(
                ecal_clusters,
                velo_tracks,
                pvs,
                name="Hlt1Pi02GammaGammaInner",
                pre_scaler_hash_string="p02gammagamma_inner_line_pre",
                post_scaler_hash_string="p02gammagamma_inner_line_post",
                enable_tupling=enable_tupling,
            ),
            make_highmass_dielectron_line(
                long_tracks,
                dileptons,
                calo_matching_objects,
                is_same_sign=True,
                pre_scaler=1.0,
                name="Hlt1DiElectronHighMass_SS",
                enable_tupling=enable_tupling,
            ),
            make_highmass_dielectron_line(
                long_tracks,
                dileptons,
                calo_matching_objects,
                is_same_sign=False,
                name="Hlt1DiElectronHighMass",
                enable_tupling=enable_tupling,
            ),
            make_di_electron_soft_line(
                long_tracks,
                dileptons_nopt,
                calo_matching_objects,
                name="Hlt1DiElectronSoft",
                enable_tupling=enable_tupling,
            ),
            make_cone_jet_line(
                jets,
                name="Hlt1ConeJet15GeV",
                min_pt=15000.0,
                pre_scaler=0.001,
                pre_scaler_hash_string="cone_jet_15gev_line_pre",
                post_scaler_hash_string="cone_jet_15gev_line_post",
                enable_tupling=enable_tupling,
            ),
            make_cone_jet_line(
                jets,
                name="Hlt1ConeJet30GeV",
                min_pt=30000.0,
                pre_scaler=0.05,
                pre_scaler_hash_string="cone_jet_30gev_line_pre",
                post_scaler_hash_string="cone_jet_30gev_line_post",
                enable_tupling=enable_tupling,
            ),
            make_cone_jet_line(
                jets,
                name="Hlt1ConeJet50GeV",
                min_pt=50000.0,
                pre_scaler=0.1,
                pre_scaler_hash_string="cone_jet_50gev_line_pre",
                post_scaler_hash_string="cone_jet_50gev_line_post",
                enable_tupling=enable_tupling,
            ),
            make_cone_jet_line(
                jets,
                name="Hlt1ConeJet100GeV",
                min_pt=100000.0,
                pre_scaler=1,
                pre_scaler_hash_string="cone_jet_100gev_line_pre",
                post_scaler_hash_string="cone_jet_100gev_line_post",
                enable_tupling=enable_tupling,
            ),
        ]

        for subSample in ["NoIP", "NoIPNorm", "Displaced"]:
            common_kwargs_low_mass_electron = {
                "minMass": 5,
                "maxMass": 300,
                "minPTprompt": 500.0,
                "minPTdisplaced": 0.0,
                "trackIPChi2Threshold": -1
                if "NoIP" in subSample
                else 4,  # it will only be picked up by the displaced line the NoIP (aka prompt) won't trigger this cut
                "selectPrompt": "NoIP" in subSample,
                "useNN": True,
                "nnCut": thresholds.DiElectronLowMassNoIP_NN
                if subSample == "NoIP"
                else thresholds.DiElectronLowMass_NN,
                "enable_monitoring": True,
                "enable_tupling": enable_tupling,
            }
            lines.append(
                make_lowmass_dielectron_line(
                    long_tracks,
                    dileptons,
                    electronid_nn,
                    calo_matching_objects,
                    **common_kwargs_low_mass_electron,
                    name="Hlt1DiElectronLowMass_{}".format(subSample),
                    pre_scaler_hash_string="Hlt1DiElectronLowMass_massSlice_{}_pre".format(
                        subSample
                    ),
                    post_scaler=0.02 if subSample == "NoIPNorm" else 1,
                )
            )
            lines.append(
                make_lowmass_dielectron_line(
                    long_tracks,
                    dileptons,
                    electronid_nn,
                    calo_matching_objects,
                    **common_kwargs_low_mass_electron,
                    is_same_sign=True,
                    name="Hlt1DiElectronLowMass_SS_{}".format(subSample),
                    pre_scaler_hash_string="Hlt1DiElectronLowMass_massSlice_SS_{}_pre".format(
                        subSample
                    ),
                    post_scaler=0.02,
                )
            )
    return [line_maker(line) for line in lines]


@configurable
def default_SMOG2_lines(
    reconstructed_objects,
    chi2_cuts,
    with_muon=True,
    with_v0s=True,
    min_z=-537.5,
    max_z=-337.5,
    enable_tupling=False,
):
    velo_tracks = reconstructed_objects["velo_tracks"]  # noqa: F841
    long_tracks = reconstructed_objects["long_tracks"]
    long_track_particles = reconstructed_objects["long_track_particles"]
    dihadrons = reconstructed_objects["dihadron_secondary_vertices"]
    prompt_dihadrons = reconstructed_objects["prompt_dihadron_secondary_vertices"]
    v0s = reconstructed_objects["v0_secondary_vertices"]
    dileptons = reconstructed_objects["dilepton_secondary_vertices"]

    lines = [
        make_SMOG2_ditrack_line(
            dihadrons,
            m1=139.57,
            m2=493.68,
            mMother=1864.83,
            mWindow=100.0,
            min_z=min_z,
            max_z=max_z,
            minEitherTrackPt=800.0,
            minTrackPt=500.0,
            minTrackIPCHI2=7.0,
            minFDCHI2=25.0,
            maxTrackChi2Ndf=chi2_cuts.Hlt1_SMOG2_DiTrack_maxTrackChi2Ndf,
            name="Hlt1SMOG2D2Kpi",
            pre_scaler=1.0,
            enable_tupling=enable_tupling,
        ),
        make_SMOG2_ditrack_line(
            prompt_dihadrons,
            m1=938.27,
            m2=938.27,
            mMother=3000.0,
            mWindow=200.0,
            min_z=min_z,
            max_z=max_z,
            minTrackIPCHI2=0.0,
            maxTrackIPCHI2=5.0,
            maxFDCHI2=20.0,
            maxTrackChi2Ndf=chi2_cuts.Hlt1_SMOG2_DiTrack_maxTrackChi2Ndf,
            minTrackP=25000.0,
            minTrackPt=1000.0,
            minEitherTrackPt=1200.0,
            name="Hlt1SMOG2etacTopp",
            pre_scaler=1.0,
            enable_tupling=enable_tupling,
        ),
        make_SMOG2_kstopipi_line(
            dihadrons,
            min_z=min_z,
            max_z=max_z,
            name="Hlt1SMOG2KsTopipi",
            minTrackPt=250.0,
            minMass=450.0,
            pre_scaler=0.3,
            enable_tupling=enable_tupling,
        ),
        make_SMOG2_ditrack_line(
            dihadrons,
            minTrackPt=500.0,
            minEitherTrackPt=800.0,
            min_z=min_z,
            max_z=max_z,
            minMdipion=1300,
            minFDCHI2=25.0,
            maxTrackChi2Ndf=chi2_cuts.Hlt1_SMOG2_DiTrack_maxTrackChi2Ndf,
            minTrackIPCHI2=7.0,
            name="Hlt1SMOG22BodyGeneric",
            enable_monitoring=False,
            enable_tupling=enable_tupling,
            pre_scaler=0.3,
        ),
        make_SMOG2_ditrack_line(
            prompt_dihadrons,
            minTrackPt=400.0,
            minEitherTrackPt=400.0,
            min_z=min_z,
            max_z=max_z,
            minTrackIPCHI2=0.0,
            maxTrackChi2Ndf=chi2_cuts.Hlt1_SMOG2_DiTrack_maxTrackChi2Ndf,
            enable_monitoring=False,
            enable_tupling=enable_tupling,
            name="Hlt1SMOG22BodyGenericPrompt",
            pre_scaler=0.01,
        ),
        make_SMOG2_singletrack_line(
            long_tracks,
            long_track_particles,
            maxChi2Ndof=chi2_cuts.Hlt1SMOG2SingleTrackVeryHighPt_maxChi2Ndof,
            name="Hlt1SMOG2SingleTrackVeryHighPt",
            minPt=5000.0,
            pre_scaler=1,
            min_z=min_z,
            max_z=max_z,
            enable_tupling=enable_tupling,
        ),
        make_SMOG2_singletrack_line(
            long_tracks,
            long_track_particles,
            maxChi2Ndof=chi2_cuts.Hlt1SMOG2SingleTrackHighPt_maxChi2Ndof,
            name="Hlt1SMOG2SingleTrackHighPt",
            minPt=3000.0,
            pre_scaler=0.1,
            min_z=min_z,
            max_z=max_z,
            enable_tupling=enable_tupling,
        ),
    ]

    if with_muon:
        muonid = reconstructed_objects["muonID"]
        lines += [
            make_SMOG2_dimuon_highmass_line(
                dileptons,
                long_tracks,
                muonid,
                maxTrackChi2Ndf=chi2_cuts.Hlt1SMOG2DiMuonHighMass_maxTrackChi2,
                maxChi2Corr=9999.0,
                enable_tupling=enable_tupling,
                name="Hlt1SMOG2DiMuonHighMass",
            ),
            make_SMOG2_single_muon_line(
                long_tracks,
                long_track_particles,
                muonid,
                maxChi2Ndof=chi2_cuts.Hlt1SMOG2SingleMuon_maxChi2Ndof,
                maxChi2Corr=1.8,
                useMuonNN=False,
                MinPt=1250,
                name="Hlt1SMOG2SingleMuon",
                pre_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            make_SMOG2_dimuon_displaced_line(
                dileptons,
                long_tracks,
                muonid,
                maxChi2Corr=1.3,
                useMuonNN=False,
                minFDCHI2=100.0,
                name="Hlt1SMOG2DisplacedDiMuon",
                enable_tupling=enable_tupling,
            ),
            make_SMOG2_jpsitomumu_tap_line(
                prompt_dihadrons,
                long_tracks,
                muonid,
                maxTrackChi2Ndf=chi2_cuts.Hlt1SMOG2JPsiToMuMuTaP_maxTrackChi2Ndf,
                posTag=True,
                maxChi2Corr=1.8,
                useMuonNN=False,
                enable_tupling=enable_tupling,
                name="Hlt1SMOG2JPsiToMuMuTaP_PosTag",
            ),
            make_SMOG2_jpsitomumu_tap_line(
                prompt_dihadrons,
                long_tracks,
                muonid,
                maxTrackChi2Ndf=chi2_cuts.Hlt1SMOG2JPsiToMuMuTaP_maxTrackChi2Ndf,
                posTag=False,
                maxChi2Corr=1.8,
                useMuonNN=False,
                enable_tupling=enable_tupling,
                name="Hlt1SMOG2JPsiToMuMuTaP_NegTag",
            ),
        ]

    if with_v0s:
        lines += [
            make_lambda2ppi_line(
                v0s,
                name="Hlt1SMOG2L0Toppi",
                minPVZ=min_z,
                maxPVZ=max_z,
                minVZ=min_z,
                maxVtxChi2=10.0,
                minDIRA=0.99985,
                minpipchi2=16.0,
                minpiipchi2=42.0,
                minpipt=150.0,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            )
        ]

    return [line_maker(line) for line in lines]


def config_SMOG2_lines(
    reconstructed_objects,
    chi2_cuts,
    odin_err_filter,
    with_calo,
    EnableGEC,
    with_muon,
    with_v0s,
    enableTupling,
):
    """Set up SMOG2-specific lines and prefilters."""
    SMOG2_prefilters = []
    SMOG2_lines = []
    SMOG2_technical_lines = []

    SMOG2_prefilters += odin_err_filter

    bx_BE = make_bxtype(bx_type=1)

    velo_closed = [
        make_event_type(
            name="ODIN_EvenType_VeloClosed", event_type="VeloOpen", invert=True
        )
    ]
    SMOG2_prefilters += velo_closed

    with line_maker.bind(prefilter=odin_err_filter + [bx_BE] + velo_closed):
        SMOG2_lines += [
            line_maker(
                make_passthrough_line(
                    name="Hlt1SMOG2BENoBias",
                    pre_scaler=3.0e-4,
                    enable_tupling=enableTupling,
                )
            )
        ]

    if with_calo:
        lowMult_5 = make_lowmult(
            reconstructed_objects["velo_tracks"],
            reconstructed_objects["ecal_clusters"],
            name="LowMult_5",
            minTracks=1,
            maxTracks=5,
        )
        with line_maker.bind(prefilter=odin_err_filter + velo_closed + [lowMult_5]):
            SMOG2_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1SMOG2PassThroughLowMult5",
                        pre_scaler=0.1,
                        enable_tupling=enableTupling,
                    )
                )
            ]

        lowMultElectrons = make_lowmult(
            reconstructed_objects["velo_tracks"],
            reconstructed_objects["ecal_clusters"],
            name="LowMultElectrons",
            minTracks=1,
            maxTracks=3,
            min_ecal_clusters=1,
            max_ecal_clusters=10,
        )
        with line_maker.bind(
            prefilter=odin_err_filter + [bx_BE] + velo_closed + [lowMultElectrons]
        ):
            SMOG2_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1SMOG2BELowMultElectrons",
                        pre_scaler=0.1,
                        enable_tupling=enableTupling,
                    )
                )
            ]

    if EnableGEC:
        gec = [
            make_gec(
                count_ut=False,
                count_velo=True,
                max_scifi_clusters=20000,
                max_velo_clusters=35000,
            )
        ]
        SMOG2_prefilters += gec

    with line_maker.bind(prefilter=SMOG2_prefilters):
        SMOG2_lines += [
            line_maker(
                make_SMOG2_minimum_bias_line(
                    reconstructed_objects["velo_tracks"],
                    reconstructed_objects["velo_states"],
                    name="Hlt1SMOG2MinimumBias",
                    pre_scaler=0.00003,
                    enable_tupling=enableTupling,
                )
            )
        ]

    SMOG2_prefilters += [
        make_checkPV(reconstructed_objects["pvs"], name="check_SMOG2_PV")
    ]

    with line_maker.bind(prefilter=SMOG2_prefilters):
        SMOG2_technical_lines += [
            line_maker(
                make_passthrough_line(
                    name="Hlt1PassthroughPVinSMOG2",
                    pre_scaler=0.00006,
                    enable_tupling=enableTupling,
                )
            )
        ]

        SMOG2_lines += default_SMOG2_lines(
            reconstructed_objects,
            chi2_cuts,
            with_muon,
            with_v0s,
            enable_tupling=enableTupling,
        )

    return SMOG2_lines, SMOG2_technical_lines, SMOG2_prefilters


# Please carefully check this function to make sure you are using the proper prefilters for the lines
def create_filter_manager(
    reconstructed_objects,
    withSMOG2=True,
    EnableGEC=True,
    with_odin_filter=True,
    velo_open=False,
    DisableLinesDuringVPClosing=True,
    tae_activity=False,
):
    """
    Creates and configures a FilterManager with named prefilters.

    Args:
        reconstructed_objects: Objects from reconstruction
        withSMOG2: Whether SMOG2 is enabled
        EnableGEC: Whether GEC is enabled
        with_odin_filter: Whether ODIN filter is enabled
        velo_open: Whether VELO is open
        DisableLinesDuringVPClosing: Whether to disable lines during VELO closing
        tae_activity: Whether TAE activity is enabled

    Returns:
        Configured FilterManager instance
    """
    # Initialize FilterManager
    preset_name = "pp_smog2_be" if withSMOG2 else "pp_default"

    filter_manager = FilterManager(
        reconstructed_objects,
        preset=preset_name,
        config_overrides={
            "prefilter_sets": {
                "base": ["odin"] if with_odin_filter else [],
                "gec": ["gec"] if EnableGEC else [],
                "bx": ["bx_BB"],
                "velo_state": ["velo_closed"] if DisableLinesDuringVPClosing else [],
            }
        },
    )

    # Define prefilter sets (PP-specific)
    filter_manager.create_named_prefilter(
        "default", ["base", "bx", "velo_state", "gec"]
    )

    filter_manager.create_named_prefilter("odin_only", ["base"])
    filter_manager.create_named_prefilter("veloMicroBias", ["odin_only"])
    filter_manager.create_named_prefilter(
        "veloMicroBias_open", ["odin_only", "velo_open"]
    )
    filter_manager.create_named_prefilter("bgi", ["base", "velo_state", "gec"])
    filter_manager.create_named_prefilter(
        "beam_gas", ["base", "bx_BE", "velo_state", "gec"]
    )
    filter_manager.create_named_prefilter("lumi", ["base", "velo_state"])
    filter_manager.create_named_prefilter(
        "tae", ["tae_activity_filter", "tae_filter"] if tae_activity else ["tae_filter"]
    )
    filter_manager.create_named_prefilter("alignment_default_prefilter", ["default"])

    return filter_manager


def setup_hlt1_node(
    enablePhysics=True,
    withMCChecking=False,
    EnableGEC=True,
    DisableLinesDuringVPClosing=True,
    withSMOG2=True,
    enableRateValidator=True,
    with_ut=True,
    with_lumi=True,
    with_odin_filter=True,
    with_calo=True,
    with_muon=True,
    with_v0s=True,
    with_rich=False,
    rich_background_method="FromReco",
    with_AC_split=False,
    enableBGI=True,
    enableAlignment=True,
    velo_open=False,
    enableDownstream=False,
    tracking_type=TrackingType.FORWARD,
    threshold_settings=get_thresholds("default"),
    tae_passthrough=True,
    nonZeroSuppress=True,
    tae_activity=False,
    enableTupling=False,
    data_quality=False,
    with_fullKF=False,
    with_ttracks=False,
    with_downstream_KF=False,
    with_quirks=True,
    passthrough_pre_scaler=0.0001,
    enabled_lines=[r".*?"],
    disabled_lines=[],
    preset_modifiers=None,
    user_hooks=False,
):
    if with_fullKF:
        from AllenConf.secondary_vertex_reconstruction import ParKF_cuts as chi2_cuts
    else:
        from AllenConf.secondary_vertex_reconstruction import (
            Velo_only_cuts as chi2_cuts,
        )

    reconstructed_objects = hlt1_reconstruction(
        with_calo=with_calo,
        with_ut=with_ut,
        with_muon=with_muon,
        enableDownstream=enableDownstream,
        tracking_type=tracking_type,
        velo_open=velo_open,
        with_AC_split=with_AC_split,
        with_rich=with_rich,
        rich_background_method=rich_background_method,
        with_fullKF=with_fullKF,
        with_ttracks=with_ttracks,
        with_downstream_KF=with_downstream_KF,
        track_max_chi2ndof=chi2_cuts.SV_track_max_chi2ndof,
    )

    preset_name = "pp_smog2_be" if withSMOG2 else "pp_default"  # noqa: F841

    filter_manager = create_filter_manager(
        reconstructed_objects=reconstructed_objects,
        withSMOG2=withSMOG2,
        EnableGEC=EnableGEC,
        with_odin_filter=with_odin_filter,
        velo_open=velo_open,
        DisableLinesDuringVPClosing=DisableLinesDuringVPClosing,
        tae_activity=tae_activity,
    )

    physics_lines = []
    smog2_lines = []
    smog2_technical_lines = []
    technical_lines = []

    if enablePhysics:
        with line_maker.bind(prefilter=filter_manager.get_prefilter_set("default")):
            physics_lines += default_physics_lines(
                reconstructed_objects,
                with_calo,
                with_muon,
                with_v0s,
                with_quirks,
                threshold_settings,
                enableTupling,
                chi2_cuts,
            )

    with line_maker.bind(prefilter=filter_manager.get_prefilter_set("odin_only")):
        technical_lines += [
            line_maker(
                make_passthrough_line(
                    pre_scaler=passthrough_pre_scaler, enable_tupling=enableTupling
                )
            )
        ]

    if tae_passthrough:
        with line_maker.bind(
            prefilter=filter_manager.get_prefilter_set("odin_only")
            + filter_manager.get_prefilter_set("tae")
        ):
            technical_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1TAEPassthrough",
                        pre_scaler=1,
                        enable_tupling=enableTupling,
                    )
                )
            ]

    if nonZeroSuppress:
        non_zero_suppress_filter = [make_nzs_filter(nzsfilter=True)]
        with line_maker.bind(
            prefilter=filter_manager.get_prefilter_set("odin_only")
            + non_zero_suppress_filter
        ):
            physics_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1NonZeroSuppress",
                        pre_scaler=1,
                        enable_tupling=enableTupling,
                    )
                )
            ]

    if EnableGEC:
        with line_maker.bind(
            prefilter=filter_manager.get_prefilter_set("odin_only")
            + filter_manager.get_prefilter_set("gec")
        ):
            technical_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1GECPassthrough", enable_tupling=enableTupling
                    )
                )
            ]

    if withSMOG2:
        smog2_lines, smog2_technical_lines, _ = config_SMOG2_lines(
            reconstructed_objects,
            chi2_cuts,
            filter_manager.get_prefilter_set("odin_only"),
            with_calo,
            EnableGEC,
            with_muon,
            with_v0s,
            enableTupling,
        )

    if preset_modifiers:
        for modifier in preset_modifiers:
            modifier()

    return setup_hlt1_base(
        reconstructed_objects=reconstructed_objects,
        filter_manager=filter_manager,
        chi2_cuts=chi2_cuts,
        physics_lines=physics_lines,
        smog2_lines=smog2_lines,
        technical_lines=technical_lines + smog2_technical_lines,
        enabled_lines=enabled_lines,
        disabled_lines=disabled_lines,
        with_lumi=with_lumi,
        with_rich=with_rich,
        enableRateValidator=enableRateValidator,
        withMCChecking=withMCChecking,
        tracking_type=tracking_type,
        with_ut=with_ut,
        with_muon=with_muon,
        with_AC_split=with_AC_split,
        with_fullKF=with_fullKF,
        enableBGI=enableBGI,
        enableAlignment=enableAlignment,
        preset="pp",
        # Additional kwargs
        EnableGEC=EnableGEC,
        DisableLinesDuringVPClosing=DisableLinesDuringVPClosing,
        user_hooks=user_hooks,
        enableTupling=enableTupling,
        data_quality=data_quality,
        with_downstream_KF=with_downstream_KF,
        with_calo=with_calo,
        reco_particles=True,
        velo_open=velo_open,
    )
