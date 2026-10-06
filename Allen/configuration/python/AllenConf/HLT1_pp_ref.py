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
from AllenConf.hlt1_charm_lines import *
from AllenConf.HLT1_common import *
from AllenConf.hlt1_downstream_lines import *
from AllenConf.hlt1_electron_lines import *
from AllenConf.hlt1_inclusive_hadron_lines import *
from AllenConf.hlt1_monitoring_lines import *
from AllenConf.hlt1_muon_lines import *
from AllenConf.hlt1_photon_lines import *
from AllenConf.hlt1_reconstruction import hlt1_reconstruction
from AllenConf.hlt1_smog2_lines import *
from AllenConf.odin import make_bxtype, make_event_type
from AllenConf.utils import line_maker


def default_physics_lines(
    reconstructed_objects,
    with_calo,
    with_muon,
    with_v0s,
    thresholds,
    enable_tupling,
    chi2_cuts,
):
    velo_tracks = reconstructed_objects["velo_tracks"]  # noqa: F841
    long_tracks = reconstructed_objects["long_tracks"]
    long_track_particles = reconstructed_objects["long_track_particles"]
    pvs = reconstructed_objects["pvs"]  # noqa: F841
    dihadrons = reconstructed_objects["dihadron_secondary_vertices"]
    dileptons = reconstructed_objects["dilepton_secondary_vertices"]
    v0s = reconstructed_objects["v0_secondary_vertices"]
    lambda_track_from_c = reconstructed_objects["lambda_track_from_c"]
    ks_track_from_c = reconstructed_objects["ks_track_from_c"]
    v0_twotrack_pairs = reconstructed_objects["v0_sv_twotrack_pairs"]
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
        ]

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
            enable_tupling=False,
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
            enable_tupling=False,
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
            pre_scaler=0.1,
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
    DisableLinesDuringVPClosing,
    smog2_lumi_prescale,
):
    """Set up SMOG2-specific lines and prefilters."""
    SMOG2_prefilters = []
    SMOG2_lines = []
    SMOG2_technical_lines = []

    SMOG2_prefilters += odin_err_filter

    bx_BE = make_bxtype(bx_type=1)

    velo_closed = (
        [
            make_event_type(
                name="ODIN_EvenType_VeloClosed", event_type="VeloOpen", invert=True
            )
        ]
        if DisableLinesDuringVPClosing
        else []
    )
    SMOG2_prefilters += velo_closed

    with line_maker.bind(prefilter=odin_err_filter + velo_closed + [bx_BE]):
        SMOG2_lines += [
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
            SMOG2_lines += [
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
            SMOG2_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1SMOG2BELowMultElectrons",
                        pre_scaler=smog2_lumi_prescale,
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
                    name="Hlt1PassthroughPVinSMOG2", pre_scaler=0.00006
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
    DisableLinesDuringVPClosing=False,
    tae_activity=False,
):
    """
    Creates and configures a FilterManager for PP/SMOG2-specific setups.

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

    # Configure the presets of alignment lines
    BGI_ACTIVITY_PRESETS["BGIPseudoPVsAll"].update(
        {
            "min_state_z": -2000,
            "max_state_z": 2000,
        }
    )
    BGI_ACTIVITY_PRESETS["BGIPseudoPVsDown"].update(
        {
            "max_state_z": 2000,
        }
    )
    BGI_ACTIVITY_PRESETS["BGIPseudoPVsUp"].update(
        {
            "min_state_z": -2000,
            "max_state_z": -250,
        }
    )
    BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsBeamTwo"]["pre_scaler"] = 1.0
    BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsDownBeamBeam"]["pre_scaler"] = 0.1
    BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsIRBeamBeam"].update(
        {"pre_scaler": 1.0, "post_scaler": 0.1}
    )
    BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsIRBeamBeam"]["enable"] = False
    ALIGNMENT_CONFIG_PRESETS["pp"]["one_muon_prescaler"] = 0.001
    ALIGNMENT_CONFIG_PRESETS["pp"]["disable_upsilon_alignment"] = True
    VELO_MICRO_BIAS_PRESETS["pp"]["Hlt1VeloMicroBiasVeloClosing"]["post_scaler"] = 0.003

    filter_manager = FilterManager(
        reconstructed_objects,
        preset=preset_name,
        config_overrides={
            "prefilter_sets": {
                "base": ["odin"] if with_odin_filter else [],
                "gec": ["gec"] if EnableGEC else [],
                "bx": ["bx_BB"],
                "velo_state": ["velo_closed"]
                if not velo_open and DisableLinesDuringVPClosing
                else [],
            }
        },
    )

    # Define prefilter sets
    filter_manager.create_named_prefilter(
        "default",
        ["base", "bx", "velo_state", "gec"]
        if EnableGEC
        else ["base", "bx", "velo_state"],
    )

    filter_manager.create_named_prefilter("odin_only", ["base"])
    filter_manager.create_named_prefilter("veloMicroBias", ["odin_only"])
    filter_manager.create_named_prefilter(
        "veloMicroBias_open", ["odin_only", "velo_open"]
    )
    filter_manager.create_named_prefilter(
        "bgi", ["base", "velo_state", "gec"] if EnableGEC else ["base", "velo_state"]
    )
    filter_manager.create_named_prefilter(
        "beam_gas",
        ["base", "bx_BE", "velo_state", "gec"]
        if EnableGEC
        else ["base", "bx_BE", "velo_state"],
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
    DisableLinesDuringVPClosing=False,
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
    enableDownstream=False,
    tracking_type=TrackingType.FORWARD,
    threshold_settings=get_thresholds("default"),
    tae_passthrough=True,
    tae_activity=False,
    enableTupling=False,
    data_quality=False,
    smog2_lumi_prescale=0.1,
    with_fullKF=False,
    with_downstream_KF=False,
    enabled_lines=[r".*?"],
    disabled_lines=[],
    preset_modifiers=None,
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
        with_fullKF=with_fullKF,
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
    technical_lines = []

    if enablePhysics:
        with line_maker.bind(prefilter=filter_manager.get_prefilter_set("default")):
            physics_lines += default_physics_lines(
                reconstructed_objects,
                with_calo,
                with_muon,
                with_v0s,
                threshold_settings,
                enableTupling,
                chi2_cuts,
            )

    with line_maker.bind(prefilter=filter_manager.get_prefilter_set("odin_only")):
        technical_lines += [line_maker(make_passthrough_line())]

    if tae_passthrough:
        with line_maker.bind(
            prefilter=filter_manager.get_prefilter_set("odin_only")
            + filter_manager.get_prefilter_set("tae")
        ):
            technical_lines += [
                line_maker(
                    make_passthrough_line(name="Hlt1TAEPassthrough", pre_scaler=1)
                )
            ]

    if EnableGEC:
        with line_maker.bind(
            prefilter=filter_manager.get_prefilter_set("odin_only")
            + filter_manager.get_prefilter_set("gec")
        ):
            technical_lines += [
                line_maker(make_passthrough_line(name="Hlt1GECPassthrough"))
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
            DisableLinesDuringVPClosing,
            smog2_lumi_prescale,
        )
        technical_lines += smog2_technical_lines

    if preset_modifiers:
        for modifier in preset_modifiers:
            modifier()

    return setup_hlt1_base(
        reconstructed_objects=reconstructed_objects,
        filter_manager=filter_manager,
        physics_lines=physics_lines,
        smog2_lines=smog2_lines,
        technical_lines=technical_lines,
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
        enableAlignment=True,
        preset="pp",
        # Additional kwargs
        EnableGEC=EnableGEC,
        velo_open=velo_open,
        DisableLinesDuringVPClosing=DisableLinesDuringVPClosing,
        user_hooks=False,
        data_quality=data_quality,
        with_downstream_KF=with_downstream_KF,
        chi2_cuts=chi2_cuts,
        with_calo=with_calo,
        with_v0s=with_v0s,
    )
