###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
import itertools

from PyConf.control_flow import CompositeNode, NodeLogic
from PyConf.tonic import configurable

from AllenConf.enum_types import TrackingType, includes_matching
from AllenConf.filters import *
from AllenConf.get_thresholds import get_thresholds
from AllenConf.hlt1_calibration_lines import *
from AllenConf.hlt1_charm_lines import *
from AllenConf.hlt1_downstream_lines import *
from AllenConf.hlt1_electron_lines import *
from AllenConf.hlt1_inclusive_hadron_lines import *
from AllenConf.hlt1_monitoring_lines import *
from AllenConf.hlt1_muon_lines import *
from AllenConf.hlt1_photon_lines import *
from AllenConf.hlt1_reconstruction import (
    hlt1_reconstruction,
    make_dq_node,
    validator_node,
)
from AllenConf.hlt1_smog2_lines import *
from AllenConf.lumi_reconstruction import lumi_reconstruction
from AllenConf.odin import (
    make_bxtype,
    make_event_type,
    make_nzs_filter,
    make_odin_orbit,
    odin_error_filter,
    tae_filter,
)
from AllenConf.persistency import make_persistency
from AllenConf.utils import line_maker, make_invert_event_list
from AllenConf.validators import rate_validation


def default_physics_lines(
    reconstructed_objects,
    with_calo,
    with_muon,
    with_v0s,
    thresholds,
    enable_tupling,
    chi2_cuts,
):
    velo_tracks = reconstructed_objects["velo_tracks"]
    long_tracks = reconstructed_objects["long_tracks"]
    long_track_particles = reconstructed_objects["long_track_particles"]
    pvs = reconstructed_objects["pvs"]
    dihadrons = reconstructed_objects["dihadron_secondary_vertices"]
    dileptons = reconstructed_objects["dilepton_secondary_vertices"]
    v0s = reconstructed_objects["v0_secondary_vertices"]
    lambda_track_from_c = reconstructed_objects["lambda_track_from_c"]  # noqa: F841
    ks_track_from_c = reconstructed_objects["ks_track_from_c"]
    v0_twotrack_pairs = reconstructed_objects["v0_sv_twotrack_pairs"]  # noqa: F841
    dstars = reconstructed_objects["dstars"]  # noqa: F841
    v0_pairs = reconstructed_objects["v0_pairs"]  # noqa: F841
    v0_hh_pairs = reconstructed_objects["v0_hh_pairs"]  # noqa: F841
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
        make_d2kpi_line(
            long_tracks,
            dihadrons,
            name="Hlt1D2KPi",
            enable_tupling=enable_tupling,
            charm_track_ip=thresholds.D2HH_track_ip,
            charm_track_pt=thresholds.D2HH_track_pt,
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
            make_downstream_two_track_ks_line(
                reconstructed_objects["downstream_tracks"],
                reconstructed_objects["downstream_secondary_vertices"],
                name="Hlt1DownstreamTwoTrackKs",
                minTrackPt_piKs=thresholds.DownstreamTwoTrackKs_minTrackPt_piKs,
                enable_monitoring=True,
                enable_tupling=enable_tupling,
            ),
        ]

    if with_v0s:
        lines += [
            make_kstopipi_line(long_tracks, v0s, name="Hlt1KsToPiPi", post_scaler=0.01),
            make_kstopipi_line(
                long_tracks,
                v0s,
                name="Hlt1KsToPiPiDoubleMuonMisID",
                double_muon_misid=True,
                enable_monitoring=True,
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
            make_ks_ll_detached_track_line(
                ks_track_from_c,
                name="Hlt1KsLLDetachedTrack",
                pi_PT_min=300,
                Ks_PT_min=700,
                h_PT_min=400,
                SUMPT_min=1000,
                enable_tupling=enable_tupling,
            ),
        ]

    if with_muon:
        muonid = reconstructed_objects["muonID"]
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
                singleMinPt=thresholds.SingleHighPtLepton_pt,
            ),
            make_di_muon_mass_line(
                long_tracks,
                dileptons,
                muonid,
                maxChi2Corr=thresholds.DiMuonHighMass_maxCorrChi2,
                name="Hlt1DiMuonHighMass",
                enable_tupling=enable_tupling,
                minHighMassTrackPt=thresholds.DiMuonHighMass_pt,
            ),
            make_di_muon_mass_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonDisplaced",
                minHighMassTrackPt=thresholds.DiMuonDisplaced_pt,
                minHighMassTrackP=3000.0,
                minMass=0.0,
                maxDoca=0.2,
                maxVertexChi2=25.0,
                minIPChi2=thresholds.DiMuonDisplaced_ipchi2,
                enable_tupling=enable_tupling,
                maxChi2Corr=thresholds.DiMuonDisplaced_maxCorrChi2,
            ),
            make_track_muon_mva_line(
                long_tracks,
                long_track_particles,
                muonid,
                maxChi2Ndof=chi2_cuts.Hlt1TrackMuonMVA_maxChi2Ndof,
                maxChi2Corr=1.8,
                name="Hlt1TrackMuonMVA",
                enable_tupling=enable_tupling,
                alpha=thresholds.TrackMuonMVA_alpha,
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
                minTrackP=10000,
                minTrackPt=1000,
                pre_scaler=0.2,
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
                minTrackP=10000,
                minTrackPt=1000,
                pre_scaler=0.2,
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
            make_cone_jet_line(
                jets,
                name="Hlt1ConeJet15GeV",
                min_pt=15000.0,
                pre_scaler=1,
                pre_scaler_hash_string="cone_jet_15gev_line_pre",
                post_scaler_hash_string="cone_jet_15gev_line_post",
            ),
            make_cone_jet_line(
                jets,
                name="Hlt1ConeJet30GeV",
                min_pt=30000.0,
                pre_scaler=1,
                pre_scaler_hash_string="cone_jet_30gev_line_pre",
                post_scaler_hash_string="cone_jet_30gev_line_post",
            ),
            make_cone_jet_line(
                jets,
                name="Hlt1ConeJet50GeV",
                min_pt=50000.0,
                pre_scaler=1,
                pre_scaler_hash_string="cone_jet_50gev_line_pre",
                post_scaler_hash_string="cone_jet_50gev_line_post",
            ),
            make_single_calo_cluster_line(
                ecal_clusters, "Hlt1HighPtPhoton", minEt=5000.0, maxEt=100000.0
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
                is_same_sign=False,
                name="Hlt1DiElectronHighMass",
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
        ]

    return [line_maker(line) for line in lines]


def odin_monitoring_lines(
    with_lumi,
    lumiline_name,
    lumilinefull_name,
    odin_err_filter,
    velo_closed_filter,
    enable_tupling=False,
):
    lines = []
    if with_lumi:
        odin_lumi_event = make_event_type(event_type="Lumi")
        with line_maker.bind(prefilter=odin_err_filter + [odin_lumi_event]):
            lines += [
                line_maker(
                    make_passthrough_line(
                        name=lumiline_name,
                        pre_scaler=1.0,
                        enable_tupling=enable_tupling,
                    )
                )
            ]

        odin_orbit = make_odin_orbit(odin_orbit_modulo=30, odin_orbit_remainder=1)
        with line_maker.bind(prefilter=odin_err_filter + [odin_lumi_event, odin_orbit]):
            lines += [
                line_maker(
                    make_passthrough_line(
                        name=lumilinefull_name,
                        pre_scaler=1.0,
                        enable_tupling=enable_tupling,
                    )
                )
            ]

    with line_maker.bind(prefilter=odin_err_filter):
        lines += [
            line_maker(
                make_odin_calib_line(
                    name="Hlt1ODINCalib", enable_tupling=enable_tupling
                )
            )
        ]

    ee_far_from_activity = make_event_type(event_type="ee_far_from_activity")
    with line_maker.bind(
        prefilter=odin_err_filter + velo_closed_filter + [ee_far_from_activity]
    ):
        lines += [
            line_maker(
                make_passthrough_line(
                    name="Hlt1ODINeeFarFromActivity",
                    pre_scaler=1.0,
                    enable_tupling=enable_tupling,
                )
            )
        ]

    return lines


def alignment_monitoring_lines(
    reconstructed_objects,
    prefilters_bx,
    chi2_cuts,
    with_muon=True,
    enable_tupling=False,
):
    long_tracks = reconstructed_objects["long_tracks"]
    long_track_particles = reconstructed_objects["long_track_particles"]
    dihadrons = reconstructed_objects["dihadron_secondary_vertices"]
    dileptons = reconstructed_objects["dilepton_secondary_vertices"]
    dstars = reconstructed_objects["dstars"]
    muon_stubs = reconstructed_objects["muon_stubs"]

    lines = [
        make_rich_1_line(
            long_tracks,
            long_track_particles,
            maxTrChi2=chi2_cuts.Hlt1RICH1Alignment_maxTrChi2,
            name="Hlt1RICH1Alignment",
            enable_tupling=enable_tupling,
        ),
        make_rich_2_line(
            long_tracks,
            long_track_particles,
            maxTrChi2=chi2_cuts.Hlt1RICH2Alignment_maxTrChi2,
            name="Hlt1RICH2Alignment",
            enable_tupling=enable_tupling,
        ),
        make_d2kpi_align_line(
            long_tracks,
            dihadrons,
            pre_scaler=0.002,
            name="Hlt1D2KPiAlignment",
            enable_tupling=enable_tupling,
        ),
        make_dst_line(
            dstars, name="Hlt1Dst2D0PiAlignment", enable_tupling=enable_tupling
        ),
    ]

    if with_muon:
        muonid = reconstructed_objects["muonID"]
        lines += [
            make_di_muon_mass_align_line(
                long_tracks,
                dileptons,
                muonid,
                name="Hlt1DiMuonJpsiMassAlignment",
                enable_tupling=enable_tupling,
            ),
            make_one_muon_track_line(
                muon_stubs["consolidated_muon_tracks"],
                muon_stubs["dev_muon_tracks_offsets"],
                muon_stubs["host_muon_total_number_of_tracks"],
                name="Hlt1OneMuonTrackLine",
                post_scaler=6e-5,
                enable_tupling=enable_tupling,
            ),
            make_di_muon_mass_align_line(
                long_tracks,
                dileptons,
                muonid,
                minHighMassTrackPt=1800.0,
                minHighMassTrackP=20000.0,
                name="Hlt1UpsilonAlignment",
                minMass=8000.0,
                maxMass=150000.0,
                minFdChi2=-1.0,
                minIP=-1.0,
                minDira=0.9,
                enable_tupling=enable_tupling,
            ),
        ]

    with line_maker.bind(prefilter=prefilters_bx):
        lines = [line_maker(line) for line in lines]

    return lines


@configurable
def velo_tomography_lines(
    reconstructed_objects,
    prefilters_odin_err,
    prefilters_bx,
    full_velo_tomography=False,
    enable_tupling=False,
):
    material_interaction_tracks = reconstructed_objects["material_interaction_tracks"]

    # VELO tomography lines need different pre-filters during special trigger configurations
    # Only apply an ODIN error filter if the full VELO tomography is enabled
    #   Otherwise it will be ODIN error + BX + VeloClosed + SciFiGEC
    tomography_prefilters = (
        prefilters_odin_err if full_velo_tomography else prefilters_bx
    )

    lines = [
        line_maker(
            make_z_range_materialvertex_seed_line(
                material_interaction_tracks,
                min_z_materialvertex_seed=300,
                max_z_materialvertex_seed=1000,
                name="Hlt1MaterialVertexSeedsDownstreamz",
                enable_tupling=enable_tupling,
            ),
            prefilter=tomography_prefilters
            + [
                make_prescaler(
                    0.5 if full_velo_tomography else 5e-4,
                    "Hlt1MaterialVertexSeedsDownstreamz",
                )
            ],
        ),
        line_maker(
            make_z_range_materialvertex_seed_line(
                material_interaction_tracks,
                min_z_materialvertex_seed=700,
                max_z_materialvertex_seed=1000,
                name="Hlt1MaterialVertexSeeds_DWFS",
                enable_tupling=enable_tupling,
            ),
            prefilter=tomography_prefilters
            + [
                make_prescaler(
                    1 if full_velo_tomography else 0.01, "Hlt1MaterialVertexSeeds_DWFS"
                )
            ],
        ),
    ]
    if full_velo_tomography:
        # Add an integrated VELO tomography line if full lines are enabled
        lines += [
            line_maker(
                make_z_range_materialvertex_seed_line(
                    material_interaction_tracks,
                    min_z_materialvertex_seed=-550,
                    max_z_materialvertex_seed=1000,
                    name="Hlt1MaterialVertexSeeds_zIntegrated",
                    enable_tupling=enable_tupling,
                ),
                prefilter=tomography_prefilters
                + [make_prescaler(1e-2, "Hlt1MaterialVertexSeeds_zIntegrated")],
            )
        ]

    return lines


@configurable
def velo_micro_bias_lines(
    reconstructed_objects,
    odin_err_filter,
    velo_micro_bias_post_scaler=1e-3,
    enable_tupling=False,
):
    velo_tracks = reconstructed_objects["velo_tracks"]
    with line_maker.bind(prefilter=odin_err_filter):
        lines = [
            line_maker(
                make_velo_micro_bias_line(
                    velo_tracks,
                    name="Hlt1VeloMicroBias",
                    pre_scaler=1.0,
                    post_scaler=velo_micro_bias_post_scaler,
                    enable_tupling=enable_tupling,
                )
            )
        ]

    velo_open_event = make_event_type(event_type="VeloOpen")
    with line_maker.bind(prefilter=odin_err_filter + [velo_open_event]):
        lines += [
            line_maker(
                make_velo_micro_bias_line(
                    velo_tracks,
                    name="Hlt1VeloMicroBiasVeloClosing",
                    post_scaler=3.0e-3,
                    enable_tupling=enable_tupling,
                )
            )
        ]

    return lines


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
            enable_monitoring=True,
            enable_tupling=enable_tupling,
            pre_scaler=1.0,
        ),
        make_SMOG2_ditrack_line(
            prompt_dihadrons,
            minTrackPt=400.0,
            minEitherTrackPt=400.0,
            min_z=min_z,
            max_z=max_z,
            minTrackIPCHI2=0.0,
            maxTrackChi2Ndf=chi2_cuts.Hlt1_SMOG2_DiTrack_maxTrackChi2Ndf,
            enable_monitoring=True,
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
        ),
        make_SMOG2_singletrack_line(
            long_tracks,
            long_track_particles,
            maxChi2Ndof=chi2_cuts.Hlt1SMOG2SingleTrackHighPt_maxChi2Ndof,
            name="Hlt1SMOG2SingleTrackHighPt",
            minPt=3000.0,
            pre_scaler=1.0,
            min_z=min_z,
            max_z=max_z,
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
                MinPt=700,
                name="Hlt1SMOG2SingleMuon",
                pre_scaler=0.2,
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


@configurable
def default_bgi_activity_lines(
    pvs,
    velo_states,
    enableBGI_full=False,
    PbPb_collision=False,
    prefilter=[],
    enable_tupling=False,
):
    """
    Primary vertex lines for various bunch crossing types composed from
    new PV filters and beam crossing lines.
    """

    mm = 1.0  # from SystemOfUnits.h
    max_cyl_rad_sq = (3 * mm) ** 2
    bx_BB = make_bxtype(bx_type=3)
    bx_NoBB = make_invert_event_list(bx_BB, name="BX_NoBeamBeam")
    pvs_z_all = make_checkCylPV(
        pvs,
        name="BGIPVsCylAll",
        min_vtx_z=-3000.0,
        max_vtx_z=3000.0,
        max_vtx_rho_sq=max_cyl_rad_sq,
        min_vtx_nTracks=10.0,
    )
    lines = []

    # Alternate version based on track beamline states
    velo_states_z_all = make_checkPseudoPV(
        velo_states,
        name="BGIPseudoPVsAll",
        min_state_z=-3000.0,
        max_state_z=3000.0,
        max_state_rho_sq=max_cyl_rad_sq,
        min_local_nTracks=10.0,
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsNoBeam",
                beam_crossing_type=0,
                pre_scaler=1.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, velo_states_z_all],
        ),
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsBeamOne",
                beam_crossing_type=1,
                pre_scaler=1.0 if enableBGI_full else 1e-2,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, velo_states_z_all],
        ),
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsBeamTwo",
                beam_crossing_type=2,
                pre_scaler=1.0 if enableBGI_full else 6.5e-2,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, velo_states_z_all],
        ),
    ]

    velo_states_z_up = make_checkPseudoPV(
        velo_states,
        name="BGIPseudoPVsUp",
        min_state_z=-3000.0,
        max_state_z=-250.0,
        max_state_rho_sq=max_cyl_rad_sq,
        min_local_nTracks=10.0,
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsUpBeamBeam",
                beam_crossing_type=3,
                pre_scaler=1.0 if enableBGI_full else 1e-3,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [velo_states_z_up],
        )
    ]

    velo_states_z_down = make_checkPseudoPV(
        velo_states,
        name="BGIPseudoPVsDown",
        min_state_z=250.0,
        max_state_z=3000.0,
        max_state_rho_sq=max_cyl_rad_sq,
        min_local_nTracks=10.0,
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsDownBeamBeam",
                beam_crossing_type=3,
                pre_scaler=1.0 if enableBGI_full else 0.05,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [velo_states_z_down],
        )
    ]

    velo_states_z_ir = make_checkPseudoPV(
        velo_states,
        name="BGIPseudoPVsIR",
        min_state_z=-250.0,
        max_state_z=250.0,
        max_state_rho_sq=max_cyl_rad_sq,
        min_local_nTracks=28.0 if not PbPb_collision else 10.0,
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsIRBeamBeam",
                beam_crossing_type=3,
                pre_scaler=0.1 if enableBGI_full else 4e-5,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [velo_states_z_ir],
        )
    ]

    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylNoBeam",
                beam_crossing_type=0,
                pre_scaler=1.0 if enableBGI_full else 0.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, pvs_z_all],
        ),
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylBeamOne",
                beam_crossing_type=1,
                pre_scaler=1.0 if enableBGI_full else 0.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, pvs_z_all],
        ),
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylBeamTwo",
                beam_crossing_type=2,
                pre_scaler=1.0 if enableBGI_full else 0.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, pvs_z_all],
        ),
    ]

    pvs_z_up = make_checkCylPV(
        pvs,
        name="BGIPVsCylUp",
        min_vtx_z=-3000.0,
        max_vtx_z=-250.0,
        max_vtx_rho_sq=max_cyl_rad_sq,
        min_vtx_nTracks=10.0,
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylUpBeamBeam",
                beam_crossing_type=3,
                pre_scaler=1.0 if enableBGI_full else 0.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [pvs_z_up],
        )
    ]

    pvs_z_down = make_checkCylPV(
        pvs,
        name="BGIPVsCylDown",
        min_vtx_z=250.0,
        max_vtx_z=3000.0,
        max_vtx_rho_sq=max_cyl_rad_sq,
        min_vtx_nTracks=10.0,
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylDownBeamBeam",
                beam_crossing_type=3,
                pre_scaler=1.0 if enableBGI_full else 0.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [pvs_z_down],
        )
    ]
    return lines


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
    with_AC_split=False,
    enableBGI=True,
    velo_open=False,
    enableDownstream=True,
    tracking_type=TrackingType.FORWARD_THEN_MATCHING,
    threshold_settings=get_thresholds("lowenergy_run_2026"),
    tae_passthrough=True,
    nonZeroSuppress=True,
    tae_activity=False,
    enableTupling=False,
    data_quality=False,
    smog2_lumi_prescale=0.1,
    smog2_mb_prescale=0.00003,
    bb_nobias_prescale=0.004,
    with_fullKF=True,
    with_downstream_KF=True,
    enabled_lines=[r".*?"],
    disabled_lines=[],
):
    if with_fullKF:
        from AllenConf.secondary_vertex_reconstruction import ParKF_cuts as chi2_cuts
    else:
        from AllenConf.secondary_vertex_reconstruction import (
            Velo_only_cuts as chi2_cuts,
        )

    hlt1_config = {}
    # Reconstruct objects needed as input for selection lines
    reconstructed_objects = hlt1_reconstruction(
        with_calo=with_calo,
        with_ut=with_ut,
        with_muon=with_muon,
        enableDownstream=enableDownstream,
        tracking_type=tracking_type,
        velo_open=velo_open,
        with_AC_split=with_AC_split,
        with_rich=with_rich,
        with_fullKF=with_fullKF,
        with_downstream_KF=with_downstream_KF,
        track_max_chi2ndof=chi2_cuts.SV_track_max_chi2ndof,
    )

    hlt1_config["reconstruction"] = reconstructed_objects
    gec = (
        [
            make_gec(
                count_ut=False,
                count_velo=True,
                max_scifi_clusters=20000,
                max_velo_clusters=35000,
            )
        ]
        if EnableGEC
        else []
    )
    odin_err_filter = (
        [odin_error_filter("odin_error_filter")] if with_odin_filter else []
    )
    beam_beam_filter = [make_bxtype(bx_type=3)]
    velo_closed = (
        [
            make_event_type(
                name="ODIN_EvenType_VeloClosed", event_type="VeloOpen", invert=True
            )
        ]
        if DisableLinesDuringVPClosing
        else []
    )

    physics_lines = []
    smog2_lines = []
    technical_lines = []

    prefilters = odin_err_filter + beam_beam_filter + velo_closed + gec
    with line_maker.bind(prefilter=prefilters):
        technical_lines += [
            line_maker(
                make_passthrough_line(
                    name="Hlt1ODIN_bbNoBias", pre_scaler=bb_nobias_prescale
                )
            )
        ]

    if enablePhysics:
        with line_maker.bind(prefilter=prefilters):
            physics_lines += default_physics_lines(
                reconstructed_objects,
                with_calo,
                with_muon,
                with_v0s,
                threshold_settings,
                enableTupling,
                chi2_cuts,
            )

    lumiline_name = "Hlt1ODINLumi"
    lumilinefull_name = "Hlt1ODIN1kHzLumi"

    technical_lines += odin_monitoring_lines(
        with_lumi,
        lumiline_name,
        lumilinefull_name,
        odin_err_filter,
        velo_closed,
        enableTupling,
    )

    with line_maker.bind(prefilter=odin_err_filter):
        technical_lines += [line_maker(make_passthrough_line())]

    # TAE passthrough
    if tae_passthrough:
        if tae_activity:
            tae_activity_filter = make_tae_activity_filter(
                reconstructed_objects["long_tracks"],
                reconstructed_objects["velo_tracks"],
            )

            tae_filters = CompositeNode(
                "taefilter_node",
                [tae_activity_filter, tae_filter()],
                NodeLogic.LAZY_AND,
                force_order=True,
            )
        else:
            tae_filters = tae_filter()

        with line_maker.bind(prefilter=odin_err_filter + [tae_filters]):
            technical_lines += [
                line_maker(
                    make_passthrough_line(name="Hlt1TAEPassthrough", pre_scaler=1)
                )
            ]

    if nonZeroSuppress:
        non_zero_suppress_filter = [make_nzs_filter(nzsfilter=True)]
        with line_maker.bind(prefilter=odin_err_filter + non_zero_suppress_filter):
            physics_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1NonZeroSuppress",
                        pre_scaler=1,
                        enable_tupling=enableTupling,
                    )
                )
            ]

    with line_maker.bind(prefilter=[sd_error_filter()]):
        technical_lines += [
            line_maker(make_passthrough_line(name="Hlt1ErrorBank", pre_scaler=0.0001))
        ]

    if EnableGEC:
        with line_maker.bind(prefilter=odin_err_filter + gec):
            technical_lines += [
                line_maker(make_passthrough_line(name="Hlt1GECPassthrough"))
            ]

    if enableBGI:
        bgi_prefilters = odin_err_filter + velo_closed + gec
        technical_lines += default_bgi_activity_lines(
            reconstructed_objects["pvs"],
            reconstructed_objects["velo_states"],
            prefilter=bgi_prefilters,
            enable_tupling=enableTupling,
        )

    technical_lines += alignment_monitoring_lines(
        reconstructed_objects,
        prefilters,
        chi2_cuts,
        with_muon,
        enable_tupling=enableTupling,
    )

    technical_lines += velo_tomography_lines(
        reconstructed_objects, odin_err_filter, prefilters, enable_tupling=enableTupling
    )

    technical_lines += velo_micro_bias_lines(
        reconstructed_objects, odin_err_filter, enable_tupling=enableTupling
    )

    with line_maker.bind(prefilter=odin_err_filter + velo_closed + gec):
        technical_lines += [
            line_maker(
                make_velo_large_clusters_line(
                    reconstructed_objects["velo_tracks"],
                    reconstructed_objects["velo_states"],
                    name="Hlt1VeloLargeClusters",
                    min_eta=5.0,
                    min_cluster_size=4,
                    min_n_hits=5,
                    enable_tupling=enableTupling,
                )
            )
        ]

    bx_BE = make_bxtype(bx_type=1)
    with line_maker.bind(prefilter=odin_err_filter + [bx_BE] + velo_closed + gec):
        technical_lines += [
            line_maker(
                make_beam_gas_line(
                    reconstructed_objects["velo_tracks"],
                    reconstructed_objects["velo_states"],
                    beam_crossing_type=1,
                    name="Hlt1BeamGas",
                )
            ),
        ]

    if withSMOG2:
        with line_maker.bind(prefilter=odin_err_filter + [bx_BE] + velo_closed):
            smog2_lines += [
                line_maker(
                    make_passthrough_line(name="Hlt1SMOG2BENoBias", pre_scaler=0.001)
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
                smog2_lines += [
                    line_maker(
                        make_passthrough_line(
                            name="Hlt1SMOG2PassThroughLowMult5", pre_scaler=0.1
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
                smog2_lines += [
                    line_maker(
                        make_passthrough_line(
                            name="Hlt1SMOG2BELowMultElectrons",
                            pre_scaler=smog2_lumi_prescale,
                        )
                    )
                ]

        SMOG2_prefilters = []
        SMOG2_prefilters += velo_closed
        if EnableGEC:
            SMOG2_prefilters += gec

        with line_maker.bind(prefilter=odin_err_filter + SMOG2_prefilters):
            smog2_lines += [
                line_maker(
                    make_SMOG2_minimum_bias_line(
                        reconstructed_objects["velo_tracks"],
                        reconstructed_objects["velo_states"],
                        name="Hlt1SMOG2MinimumBias",
                        pre_scaler=smog2_mb_prescale,
                    )
                )
            ]

        SMOG2_prefilters += [
            make_checkPV(reconstructed_objects["pvs"], name="check_SMOG2_PV")
        ]

        with line_maker.bind(prefilter=odin_err_filter + SMOG2_prefilters):
            technical_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1PassthroughPVinSMOG2", pre_scaler=0.00006
                    )
                )
            ]

            smog2_lines += default_SMOG2_lines(
                reconstructed_objects,
                chi2_cuts,
                with_muon,
                with_v0s,
                enable_tupling=enableTupling,
            )

    grouped_lines = dict(
        Physics=physics_lines,
        SMOG2=smog2_lines,
        Technical=technical_lines,
    )
    grouped_line_algs = {
        key: [tup[0] for tup in lines] for key, lines in grouped_lines.items()
    }
    grouped_line_algs = {
        key: regex_filter_lines(line_algs, enabled_lines, disabled_lines)
        for key, line_algs in grouped_line_algs.items()
    }
    line_algorithms = [alg for alg in itertools.chain(*grouped_line_algs.values())]

    line_nodes = [tup[1] for tup in itertools.chain(*grouped_lines.values())]
    lines = CompositeNode(
        "SetupAllLines", line_nodes, NodeLogic.NONLAZY_OR, force_order=False
    )

    persistency_node, persistency_algorithms = make_persistency(line_algorithms)

    hlt1_node = CompositeNode(
        "Allen", [lines, persistency_node], NodeLogic.NONLAZY_AND, force_order=True
    )

    hlt1_config["line_nodes"] = line_nodes
    hlt1_config["line_algorithms"] = line_algorithms
    hlt1_config.update(persistency_algorithms)

    if with_lumi:
        lumi_reco = lumi_reconstruction(
            gather_selections=hlt1_config["gather_selections"],
            lumiline_name=lumiline_name,
            lumilinefull_name=lumilinefull_name,
            with_muon=with_muon,
            velo_open=velo_open,
        )

        lumi_node = CompositeNode(
            "AllenLumiNode",
            lumi_reco["algorithms"],
            NodeLogic.NONLAZY_AND,
            force_order=False,
        )

        lumi_with_prefilter = CompositeNode(
            "LumiWithPrefilter",
            odin_err_filter + velo_closed + [lumi_node],
            NodeLogic.LAZY_AND,
            force_order=True,
        )

        hlt1_config["lumi_reconstruction"] = lumi_reco
        hlt1_config["lumi_node"] = lumi_with_prefilter

        hlt1_node = CompositeNode(
            "AllenWithLumi",
            [hlt1_node, lumi_with_prefilter],
            NodeLogic.NONLAZY_AND,
            force_order=False,
        )

    if enableRateValidator:
        hlt1_node = CompositeNode(
            "AllenRateValidation",
            [
                hlt1_node,
                rate_validation(lines=line_algorithms, groups=grouped_line_algs),
            ],
            NodeLogic.NONLAZY_AND,
            force_order=True,
        )
    if data_quality:
        # Forward reconstructed long tracks are needed for that module
        # Matching method is already used in reconstructed_objects
        reconstructed_objects_forward = hlt1_reconstruction(
            "ODQV_forward",
            with_calo=with_calo,
            with_ut=with_ut,
            with_muon=with_muon,
            tracking_type=TrackingType.FORWARD,
        )
        node = make_dq_node(
            reconstructed_objects, reconstructed_objects_forward, line_algorithms
        )
        return node

    if not withMCChecking:
        hlt1_config["control_flow_node"] = hlt1_node
    else:
        validation_node = validator_node(
            reconstructed_objects,
            line_algorithms,
            includes_matching(tracking_type),
            with_ut,
            with_muon,
            with_AC_split,
            with_fullKF,
            with_downstream_KF,
            prefilters,
        )
        hlt1_config["validator_node"] = validation_node

        node = CompositeNode(
            "AllenWithValidators",
            [hlt1_node, validation_node],
            NodeLogic.NONLAZY_AND,
            force_order=False,
        )
        hlt1_config["control_flow_node"] = node

    return hlt1_config
