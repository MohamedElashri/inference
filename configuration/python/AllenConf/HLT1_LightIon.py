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
from AllenConf.enum_types import ActivityType, TrackingType
from AllenConf.filters import *
from AllenConf.hlt1_calibration_lines import *
from AllenConf.hlt1_charm_lines import *
from AllenConf.HLT1_common import *
from AllenConf.hlt1_heavy_ions_lines import *
from AllenConf.hlt1_inclusive_hadron_lines import *
from AllenConf.hlt1_monitoring_lines import *
from AllenConf.hlt1_muon_lines import *
from AllenConf.hlt1_reconstruction import hlt1_reconstruction
from AllenConf.utils import line_maker


def default_physics_lines(
    reconstructed_objects, prescale, reco_particles, with_muon, chi2_cuts
):
    velo_tracks = reconstructed_objects["velo_tracks"]
    long_tracks = reconstructed_objects["long_tracks"]
    long_track_particles = reconstructed_objects["long_track_particles"]
    decoded_calo = reconstructed_objects["decoded_calo"]
    pvs = reconstructed_objects["pvs"]
    dihadrons = reconstructed_objects["dihadron_secondary_vertices"]
    dileptons = reconstructed_objects["dilepton_secondary_vertices"]
    v0s = reconstructed_objects["v0_secondary_vertices"]
    muon_stubs = reconstructed_objects["muon_stubs"]  # noqa: F841

    physics_lines = [
        make_heavy_ion_event_line(
            name="Hlt1LightIonMicroBias",
            velo_tracks=velo_tracks,
            long_track_particles=long_track_particles,
            pvs=pvs,
            decoded_calo=decoded_calo,
            min_velo_tracks_PbPb=1,
            pre_scaler=0.5 if prescale else 1,
        ),
    ]
    smog2_lines = [
        make_heavy_ion_event_line(
            name="Hlt1LightIonSMOGMicroBias",
            velo_tracks=velo_tracks,
            long_track_particles=long_track_particles,
            pvs=pvs,
            decoded_calo=decoded_calo,
            min_velo_tracks_SMOG=1,
            pre_scaler=0.5 if prescale else 1,
        ),
    ]
    if reco_particles:
        physics_lines += [
            make_kstopipi_line(
                long_tracks, v0s, name="Hlt1KsToPiPi", post_scaler=0.001
            ),
            make_kstopipi_line(
                long_tracks,
                v0s,
                name="Hlt1KsToPiPiDoubleMuonMisID",
                double_muon_misid=True,
            ),
            make_d2kk_line(long_tracks, dihadrons, name="Hlt1D2KK"),
            make_d2kpi_line(long_tracks, dihadrons, name="Hlt1D2KPi"),
            make_d2pipi_line(long_tracks, dihadrons, name="Hlt1D2PiPi"),
            make_lambda2ppi_line(v0s, name="Hlt1L02PPi"),
        ]
        if with_muon:
            muonid = reconstructed_objects["muonID"]
            physics_lines += [
                make_di_muon_mass_line(
                    long_tracks, dileptons, muonid, name="Hlt1DiMuonHighMass"
                ),
                make_di_muon_mass_line(
                    long_tracks,
                    dileptons,
                    muonid,
                    name="Hlt1DiMuonLowMass",
                    enable_monitoring=False,
                    minHighMassTrackPt=500.0,
                    minHighMassTrackP=3000.0,
                    minMass=0.0,
                    maxDoca=0.2,
                    maxVertexChi2=25.0,
                    minIPChi2=4.0,
                ),
            ]

    return [line_maker(line) for line in physics_lines], [
        line_maker(line) for line in smog2_lines
    ]


def upc_physics_lines(reconstructed_objects):
    pvs = reconstructed_objects["pvs"]
    velo_tracks = reconstructed_objects["velo_tracks"]
    ecal_clusters = reconstructed_objects["ecal_clusters"]

    lines = [
        make_diphoton_lowmult_line(
            name="Hlt1LightIonUPCDiPhoton_LowPt_Ycut",
            calo=ecal_clusters,
            velo_tracks=velo_tracks,
            pvs=pvs,
            min_absY=100,
            max_velo_tracks=10,
            max_ecal_clusters=10,
            maxPt=1000,
        ),
        make_photon_lowmult_line(
            name="Hlt1LightIonUPCPhoton_Ycut",
            calo=ecal_clusters,
            min_absY=100,
            max_ecal_clusters=10,
            pre_scaler=0.02,
        ),
        make_photon_lowmult_line(
            name="Hlt1LightIonUPCPhoton",
            calo=ecal_clusters,
            max_ecal_clusters=10,
            pre_scaler=0.002,
        ),
        make_diphoton_lowmult_line(
            name="Hlt1LightIonUPCDiPhoton_HighMass",
            calo=ecal_clusters,
            velo_tracks=velo_tracks,
            pvs=pvs,
            minMass=1300,
            minEt_clusters=500,
            maxPt=2000,
            max_velo_tracks=10,
            max_ecal_clusters=10,
            mass_histogram_range=[1300, 40000],
            pre_scaler=0.1,
        ),
        make_photon_lowmult_line(
            name="Hlt1LightIonUPCPhoton_HighEt",
            calo=ecal_clusters,
            minEt=800,
            max_ecal_clusters=10,
            pre_scaler=0.1,
        ),
    ]
    return [line_maker(line) for line in lines]


def mini_physics_lines(reconstructed_objects):
    velo_tracks = reconstructed_objects["velo_tracks"]
    long_track_particles = reconstructed_objects["long_track_particles"]
    decoded_calo = reconstructed_objects["decoded_calo"]
    pvs = reconstructed_objects["pvs"]
    ecal_clusters = reconstructed_objects["ecal_clusters"]

    lines = [
        make_heavy_ion_event_line(
            name="Hlt1HeavyIonLightIonUPCMB",
            velo_tracks=velo_tracks,
            long_track_particles=long_track_particles,
            pvs=pvs,
            decoded_calo=decoded_calo,
            max_ecal_e=94000,
            min_long_tracks=1,
            min_velo_tracks_PbPb=2,
            pre_scaler=1,
        ),
        make_photon_lowmult_line(
            name="Hlt1LightIonUPCPhoton",
            pre_scaler_hash_string="LightIonUPCPhoton_line_pre",
            post_scaler_hash_string="LightIonUPCPhoton_line_post",
            calo=ecal_clusters,
            max_ecal_clusters=10,
        ),
    ]

    return [line_maker(line) for line in lines]


# Please carefully check this function to make sure you are using the proper prefilters for the lines
def create_filter_manager(
    reconstructed_objects,
    mini=False,
    EnableGEC=False,
    with_odin_filter=True,
    DisableLinesDuringVPClosing=False,
    ActivityForClosing=ActivityType.VELO_CLUSTERS,
    max_ecal_upc=94000,
    min_ecal_hadro=94000,
    minimal_activity_type=ActivityType.VELO_CLUSTERS,
    velo_open=False,
    tae_activity=True,
):
    """
    Creates and configures a FilterManager for LightIon-specific setups.

    Args:
        reconstructed_objects: Objects from reconstruction
        mini: Whether to use mini configuration (UPC-focused)
        EnableGEC: Whether GEC is enabled
        with_odin_filter: Whether ODIN filter is enabled
        DisableLinesDuringVPClosing: Whether to disable lines during VELO closing
        ActivityForClosing: Type of activity to use for VELO closing
        max_ecal_upc: Maximum ECAL energy for UPC
        min_ecal_hadro: Minimum ECAL energy for hadronic
        minimal_activity_type: Type of activity for minimal activity filter
        velo_open: Whether VELO is open
        tae_activity: Whether TAE activity is enabled

    Returns:
        Configured FilterManager instance
    """
    # Initialize FilterManager
    preset_name = "LightIon_mini" if mini else "LightIon_default"

    # Configure the presets of alignment lines
    BGI_ACTIVITY_PRESETS["BGIPseudoPVsIR"]["min_local_nTracks"] = 10

    # Create config overrides based on parameters
    config_overrides = {
        "parameters": {
            "gec_upc": {
                "ecal_cut": max_ecal_upc,
            },
            "gec_hadronic": {
                "min_ecal_hadro": min_ecal_hadro,
            },
            "velo_closing_filter": {"max_clusters": 5000},
            "activity_type": minimal_activity_type,
            "activity_type_closing": ActivityForClosing,
        },
        "prefilter_sets": {
            "base": ["odin"] if with_odin_filter else [],
            "gec": ["gec"] if EnableGEC else [],
            "velo_state": ["velo_closed"] if DisableLinesDuringVPClosing else [],
            "velo_closing_gec": (
                ["velo_closing_filter"]
                if ActivityForClosing == ActivityType.VELO_CLUSTERS
                else ["pv_activity_filter"]
            ),
            "veloMicroBias_gec": (
                ["veloMicroBias_clusters_filter"]
                if ActivityForClosing == ActivityType.VELO_CLUSTERS
                else ["pv_activity_filter"]
            ),
        },
    }

    filter_manager = FilterManager(
        reconstructed_objects, preset=preset_name, config_overrides=config_overrides
    )

    # Define prefilter sets (LightIon-specific)

    # Default prefilter set
    filter_manager.create_named_prefilter("default", ["base", "gec", "velo_state"])

    # UPC prefilter set (for mini mode and UPC physics)
    filter_manager.create_named_prefilter("upc", ["default", "gec_upc", "velo_state"])

    # Hadronic prefilter set
    filter_manager.create_named_prefilter(
        "hadronic", ["base", "gec_hadronic", "velo_state"]
    )

    # Lumi prefilter set (different for mini vs regular)
    filter_manager.create_named_prefilter(
        "lumi", ["base", "gec_upc"] if mini else ["base", "gec"]
    )

    # BGI prefilter sets
    filter_manager.create_named_prefilter(
        "bgi", ["base", "gec_upc"] if mini else ["base", "gec"]
    )
    filter_manager.create_named_prefilter(
        "beam_gas", ["upc" if mini else "default", "bx_BE"]
    )

    # ODIN-only prefilter set
    filter_manager.create_named_prefilter("odin_only", ["base"])

    # Velo micro-bias prefilter sets
    filter_manager.create_named_prefilter(
        "veloMicroBias", ["base", "bx_BB", "velo_state", "veloMicroBias_gec"]
    )

    # Velo micro-bias during Velo open
    filter_manager.create_named_prefilter(
        "veloMicroBias_open", ["base", "velo_open", "gec", "velo_closing_gec"]
    )

    # Photon+Velo UPC prefilter set
    filter_manager.create_named_prefilter(
        "photon_velo_upc", ["base", "gec", "velo_state", "gec_photon_nvelo_upc"]
    )

    # Minimal activity prefilter set
    filter_manager.create_named_prefilter(
        "Hlt1MinimalActivity", ["base", "activity_filter"]
    )

    # TAE prefilter set
    filter_manager.create_named_prefilter(
        "tae",
        ["gec", "pv_activity_filter", "tae_filter"] if tae_activity else ["tae_filter"],
    )
    filter_manager.create_named_prefilter(
        "alignment_default_prefilter", ["upc"] if mini else ["default"]
    )

    return filter_manager


@configurable
def setup_hlt1_node(
    withMCChecking=False,
    max_ecal_upc=94000,
    min_ecal_hadro=94000,
    EnableGEC=False,
    enableBGI=True,
    enableRateValidator=True,
    with_lumi=True,
    with_odin_filter=True,
    tracking_type=TrackingType.FORWARD,
    with_ut=True,
    with_AC_split=False,
    prescale=False,
    with_calo=True,
    with_muon=True,
    with_rich=False,
    velo_open=False,
    enableDownstream=True,
    reco_particles=True,
    bx_type=None,
    tae_passthrough=True,
    tae_activity=True,
    minimal_activity_type=ActivityType.VELO_CLUSTERS,
    ActivityForClosing=ActivityType.VELO_CLUSTERS,
    DisableLinesDuringVPClosing=False,
    mini=False,
    with_fullKF=False,
    with_downstream_KF=False,
    enabled_lines=[r".*?"],
    disabled_lines=[],
    preset_modifiers=None,
):
    """
    Setup HLT1 node for LightIon runs using the common infrastructure.

    Args:
        mini: If True, use mini configuration (UPC-focused)
        with_fullKF: If True, use full Kalman Filter for secondary vertices
        ActivityForClosing: Activity type to use for Velo closing monitoring
        DisableLinesDuringVPClosing: If True, disable lines during Velo closing
        tae_passthrough: If True, include TAE passthrough line
        tae_activity: If True, include activity filter with TAE
        bx_type: Beam crossing type filter (None for default)
        enabled_lines: List of regex patterns for enabled lines
        disabled_lines: List of regex patterns for disabled lines
    """

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

    preset_name = "LightIon_mini" if mini else "LightIon_default"  # noqa: F841

    filter_manager = create_filter_manager(
        reconstructed_objects=reconstructed_objects,
        mini=mini,
        EnableGEC=EnableGEC,
        with_odin_filter=with_odin_filter,
        DisableLinesDuringVPClosing=DisableLinesDuringVPClosing,
        ActivityForClosing=ActivityForClosing,
        max_ecal_upc=max_ecal_upc,
        min_ecal_hadro=min_ecal_hadro,
        minimal_activity_type=minimal_activity_type,
        velo_open=velo_open,
        tae_activity=tae_activity,
    )

    physics_lines = []
    smog2_lines = []
    technical_lines = []

    if mini:
        # Mini mode: UPC-focused physics
        with line_maker.bind(prefilter=filter_manager.get_prefilter_set("upc")):
            physics_lines = mini_physics_lines(reconstructed_objects)
            physics_lines += [
                line_maker(make_passthrough_line(name="Hlt1GECUPCPassthrough"))
            ]

        with line_maker.bind(prefilter=filter_manager.get_prefilter_set("hadronic")):
            physics_lines += [
                line_maker(
                    make_passthrough_line(name="Hlt1GECCentPassthrough", pre_scaler=1)
                )
            ]

        with line_maker.bind(prefilter=filter_manager.get_prefilter_set("default")):
            physics_lines += [
                line_maker(make_passthrough_line(name="Hlt1GECSciFiPassthrough"))
            ]
    else:
        # Regular mode: Full physics program
        default_prefilters = filter_manager.get_prefilter_set("default")

        # Add BX type filter if specified
        if bx_type is not None:
            default_prefilters = default_prefilters + filter_manager.with_bx_type(
                "default", bx_type
            )

        with line_maker.bind(prefilter=default_prefilters):
            physics_lines, smog2_lines = default_physics_lines(
                reconstructed_objects, prescale, reco_particles, with_muon, chi2_cuts
            )

        # Add UPC physics lines
        with line_maker.bind(
            prefilter=filter_manager.get_prefilter_set("photon_velo_upc")
        ):
            physics_lines += upc_physics_lines(reconstructed_objects)

            # Add GEC passthrough if GEC is enabled
            if EnableGEC:
                technical_lines += [
                    line_maker(make_passthrough_line(name="Hlt1GECPassthrough"))
                ]

    with line_maker.bind(prefilter=filter_manager.get_prefilter_set("odin_only")):
        technical_lines += [line_maker(make_passthrough_line())]

    if tae_passthrough:
        tae_prefilters = filter_manager.get_prefilter_set(
            "odin_only"
        ) + filter_manager.get_prefilter_set("tae")
        with line_maker.bind(prefilter=tae_prefilters):
            technical_lines += [
                line_maker(
                    make_passthrough_line(name="Hlt1TAEPassthrough", pre_scaler=1)
                )
            ]
    if preset_modifiers:
        for modifier in preset_modifiers:
            modifier()

    return setup_hlt1_base(
        reconstructed_objects=reconstructed_objects,
        filter_manager=filter_manager,
        chi2_cuts=chi2_cuts,
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
        enableAlignment=True,  # Alignment monitoring typically enabled for PbPb
        preset="LightIon_mini" if mini else "LightIon",  # Pass appropriate preset
        # Additional configuration passed to monitoring lines
        EnableGEC=EnableGEC,
        velo_open=velo_open,
        DisableLinesDuringVPClosing=DisableLinesDuringVPClosing,
        with_calo=with_calo,
        with_downstream_KF=with_downstream_KF,
        reco_particles=reco_particles,
        # BGI specific
        # For monitoring lines context
        mini=mini,
    )
