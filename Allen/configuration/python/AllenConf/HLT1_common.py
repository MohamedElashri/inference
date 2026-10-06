###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "COPYING".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
import itertools

from PyConf.control_flow import CompositeNode, NodeLogic
from PyConf.tonic import configurable

from AllenConf.enum_types import TrackingType, includes_matching
from AllenConf.filtermanager import *
from AllenConf.hlt1_calibration_lines import *
from AllenConf.hlt1_monitoring_lines import *
from AllenConf.hlt1_muon_lines import *
from AllenConf.hlt1_presets import *
from AllenConf.hlt1_reconstruction import (
    hlt1_reconstruction,
    make_dq_node,
    validator_node,
)
from AllenConf.lumi_reconstruction import lumi_reconstruction
from AllenConf.odin import make_bxtype, make_event_type, make_odin_orbit
from AllenConf.persistency import make_persistency
from AllenConf.utils import line_maker, make_invert_event_list
from AllenConf.validators import rate_validation


@configurable
def default_bgi_activity_lines(
    pvs,
    velo_states,
    enableBGI_full=False,
    enable_tupling=False,
    preset="pp",
    prefilter=[],
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
        min_vtx_z=BGI_ACTIVITY_PRESETS["BGIPVsCylAll"]["min_vtx_z"],
        max_vtx_z=BGI_ACTIVITY_PRESETS["BGIPVsCylAll"]["max_vtx_z"],
        max_vtx_rho_sq=max_cyl_rad_sq,
        min_vtx_nTracks=BGI_ACTIVITY_PRESETS["BGIPVsCylAll"]["min_vtx_nTracks"],
    )
    lines = []

    velo_states_z_all = make_checkPseudoPV(
        velo_states,
        name="BGIPseudoPVsAll",
        min_state_z=BGI_ACTIVITY_PRESETS["BGIPseudoPVsAll"]["min_state_z"],
        max_state_z=BGI_ACTIVITY_PRESETS["BGIPseudoPVsAll"]["max_state_z"],
        max_state_rho_sq=max_cyl_rad_sq,
        min_local_nTracks=BGI_ACTIVITY_PRESETS["BGIPseudoPVsAll"]["min_local_nTracks"],
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
                pre_scaler=1.0
                if enableBGI_full
                else BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsBeamOne"]["pre_scaler"],
                post_scaler=BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsBeamOne"][
                    "post_scaler"
                ],
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, velo_states_z_all],
        ),
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsBeamTwo",
                beam_crossing_type=2,
                pre_scaler=1.0
                if enableBGI_full
                else BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsBeamTwo"]["pre_scaler"],
                post_scaler=BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsBeamTwo"][
                    "post_scaler"
                ],
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, velo_states_z_all],
        ),
    ]

    velo_states_z_up = make_checkPseudoPV(
        velo_states,
        name="BGIPseudoPVsUp",
        min_state_z=BGI_ACTIVITY_PRESETS["BGIPseudoPVsUp"]["min_state_z"],
        max_state_z=BGI_ACTIVITY_PRESETS["BGIPseudoPVsUp"]["max_state_z"],
        max_state_rho_sq=max_cyl_rad_sq,
        min_local_nTracks=BGI_ACTIVITY_PRESETS["BGIPseudoPVsUp"]["min_local_nTracks"],
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsUpBeamBeam",
                beam_crossing_type=3,
                pre_scaler=1.0
                if enableBGI_full
                else BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsUpBeamBeam"]["pre_scaler"],
                post_scaler=BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsUpBeamBeam"][
                    "post_scaler"
                ],
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [velo_states_z_up],
        )
    ]

    velo_states_z_down = make_checkPseudoPV(
        velo_states,
        name="BGIPseudoPVsDown",
        min_state_z=BGI_ACTIVITY_PRESETS["BGIPseudoPVsDown"]["min_state_z"],
        max_state_z=BGI_ACTIVITY_PRESETS["BGIPseudoPVsDown"]["max_state_z"],
        max_state_rho_sq=max_cyl_rad_sq,
        min_local_nTracks=BGI_ACTIVITY_PRESETS["BGIPseudoPVsDown"]["min_local_nTracks"],
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPseudoPVsDownBeamBeam",
                beam_crossing_type=3,
                pre_scaler=1.0
                if enableBGI_full
                else BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsDownBeamBeam"]["pre_scaler"],
                post_scaler=BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsDownBeamBeam"][
                    "post_scaler"
                ],
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [velo_states_z_down],
        )
    ]

    velo_states_z_ir = make_checkPseudoPV(
        velo_states,
        name="BGIPseudoPVsIR",
        min_state_z=BGI_ACTIVITY_PRESETS["BGIPseudoPVsIR"]["min_state_z"],
        max_state_z=BGI_ACTIVITY_PRESETS["BGIPseudoPVsIR"]["max_state_z"],
        max_state_rho_sq=max_cyl_rad_sq,
        min_local_nTracks=BGI_ACTIVITY_PRESETS["BGIPseudoPVsIR"]["min_local_nTracks"],
    )
    if BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsIRBeamBeam"]["enable"]:
        lines += [
            line_maker(
                make_beam_line(
                    name="Hlt1BGIPseudoPVsIRBeamBeam",
                    beam_crossing_type=3,
                    pre_scaler=1.0
                    if enableBGI_full
                    else BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsIRBeamBeam"][
                        "pre_scaler"
                    ],
                    post_scaler=BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsIRBeamBeam"][
                        "post_scaler"
                    ],
                    enable_tupling=enable_tupling,
                ),
                prefilter=prefilter + [velo_states_z_ir],
            )
        ]

    # FIXME:the place of this line mismatches between HLT1_pp.py and HLT1_pp_ref.py. I assume the following one is correct.
    if not enableBGI_full:
        return lines

    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylNoBeam",
                beam_crossing_type=0,
                pre_scaler=1.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, pvs_z_all],
        ),
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylBeamOne",
                beam_crossing_type=1,
                pre_scaler=1.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, pvs_z_all],
        ),
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylBeamTwo",
                beam_crossing_type=2,
                pre_scaler=1.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [bx_NoBB, pvs_z_all],
        ),
    ]

    pvs_z_up = make_checkCylPV(
        pvs,
        name="BGIPVsCylUp",
        min_vtx_z=BGI_ACTIVITY_PRESETS["BGIPVsCylAll"]["min_vtx_z"],
        max_vtx_z=-250.0,
        max_vtx_rho_sq=max_cyl_rad_sq,
        min_vtx_nTracks=10.0,
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylUpBeamBeam",
                beam_crossing_type=3,
                pre_scaler=1.0,
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
        max_vtx_z=BGI_ACTIVITY_PRESETS["BGIPVsCylAll"]["max_vtx_z"],
        max_vtx_rho_sq=max_cyl_rad_sq,
        min_vtx_nTracks=10.0,
    )
    lines += [
        line_maker(
            make_beam_line(
                name="Hlt1BGIPVsCylDownBeamBeam",
                beam_crossing_type=3,
                pre_scaler=1.0,
                post_scaler=1.0,
                enable_tupling=enable_tupling,
            ),
            prefilter=prefilter + [pvs_z_down],
        )
    ]
    if BGI_ACTIVITY_PRESETS["BGIPVsCylIR"]["enable"]:
        pvs_z_ir = make_checkCylPV(
            pvs,
            name="BGIPVsCylIR",
            min_vtx_z=-250.0,
            max_vtx_z=BGI_ACTIVITY_PRESETS["BGIPVsCylIR"]["max_vtx_z"],
            max_vtx_rho_sq=max_cyl_rad_sq,
            min_vtx_nTracks=28.0,
        )
        lines += [
            line_maker(
                make_beam_line(
                    name="Hlt1BGIPVsCylIRBeamBeam",
                    beam_crossing_type=3,
                    pre_scaler=1.0,
                    post_scaler=1.0,
                ),
                prefilter=prefilter + [pvs_z_ir],
            )
        ]

    return lines


def config_odin_monitoring_lines(
    with_lumi=True,
    with_gec=False,
    with_ee_far=False,
    lumiline_name="Hlt1ODINLumi",
    lumilinefull_name="Hlt1ODIN1kHzLumi",
    odin_err_filter=None,
    velo_closed_filter=None,
    enable_tupling=False,
):
    lines = []

    if with_lumi:
        odin_lumi_event = make_event_type(event_type="Lumi")
        if with_gec:
            with line_maker.bind(prefilter=odin_err_filter):
                lines.append(
                    line_maker(
                        make_odin_event_type_with_decoding_line(
                            name=lumiline_name, odin_event_type="Lumi"
                        )
                    )
                )
        else:
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

    if with_ee_far:
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


def config_alignment_monitoring_lines(
    reconstructed_objects,
    prefilters,
    chi2_cuts,
    with_muon=True,
    enable_tupling=False,
    reco_particles=True,
    preset="pp",
):
    """
    Unified alignment monitoring lines function.
    Use preset='pp' for pp data taking, preset='PbPb', 'LightIon' for Heavy/Light Ion data taking.
    """

    config = ALIGNMENT_CONFIG_PRESETS[preset]

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
    ]

    if reco_particles:
        lines += [
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

        muon_lines = [
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
                post_scaler=config["one_muon_prescaler"],
                enable_tupling=enable_tupling,
            ),
        ]

        # FIXME: Upsilon alignment - different implementations. can be unified?
        if not config["disable_upsilon_alignment"]:
            upsilon_cfg = config["upsilon_alignment"]
            if preset == "pp":
                muon_lines.append(
                    make_di_muon_mass_align_line(
                        long_tracks,
                        dileptons,
                        muonid,
                        name="Hlt1UpsilonAlignment",
                        enable_tupling=enable_tupling,
                        **upsilon_cfg,
                    )
                )
            elif preset in ["PbPb", "LightIon"]:
                muon_lines.append(
                    make_di_muon_mass_line(
                        long_tracks,
                        dileptons,
                        muonid,
                        name="Hlt1UpsilonAlignment",
                        **upsilon_cfg,
                    )
                )

        if config["include_jpsi_tagging"]:
            muon_lines.extend(
                [
                    make_det_jpsitomumu_tap_line(
                        long_tracks,
                        dihadrons,
                        name="Hlt1DetJpsiToMuMuPosTagLine",
                        posTag=True,
                        enable_monitoring=True,
                        enable_tupling=False,
                    ),
                    make_det_jpsitomumu_tap_line(
                        long_tracks,
                        dihadrons,
                        name="Hlt1DetJpsiToMuMuNegTagLine",
                        posTag=False,
                        enable_monitoring=True,
                        enable_tupling=False,
                    ),
                ]
            )

        lines += muon_lines

    lines = [line_maker(line, prefilter=prefilters) for line in lines]

    return lines


@configurable
def config_velo_large_clusters_lines(
    reconstructed_objects, prefilters=None, enable_tupling=False, preset="pp"
):
    """
    Unified VELO large cluster lines function.
    """
    config = VELO_LARGE_CLUSTERS_CONFIG_PRESETS[preset]
    lines = []
    if config["enable"]:
        with line_maker.bind(prefilter=prefilters):
            lines.append(
                line_maker(
                    make_velo_large_clusters_line(
                        reconstructed_objects["velo_tracks"],
                        reconstructed_objects["velo_states"],
                        name="Hlt1VeloLargeClusters",
                        min_eta=config["min_eta"],
                        min_cluster_size=config["min_cluster_size"],
                        min_n_hits=config["min_n_hits"],
                        enable_tupling=enable_tupling,
                    )
                )
            )
    return lines


@configurable
def config_velo_tomography_lines(
    reconstructed_objects,
    prefilters_odin_err=None,
    prefilters_bx=None,
    enable_tupling=False,
    preset="pp",
):
    """
    Unified VELO tomography lines function.
    """

    config = VELO_TOMOGRAPHY_CONFIG_PRESETS[preset]

    material_interaction_tracks = reconstructed_objects.get(
        "material_interaction_tracks"
    )
    if material_interaction_tracks is None:
        return []

    tomography_prefilters = (
        prefilters_odin_err if config["full_velo_tomography"] else prefilters_bx
    )

    if config["add_velo_gec"]:
        material_interaction_velo_gec = velo_gec("material_velo_gec", 0, 35000)
        tomography_prefilters = list(tomography_prefilters) + [
            material_interaction_velo_gec
        ]

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
                    config["prescalers"]["downstreamz_full"]
                    if config["full_velo_tomography"]
                    else config["prescalers"]["downstreamz"],
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
                    config["prescalers"]["dwfs_full"]
                    if config["full_velo_tomography"]
                    else config["prescalers"]["dwfs"],
                    "Hlt1MaterialVertexSeeds_DWFS",
                )
            ],
        ),
    ]

    if config["full_velo_tomography"]:
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


def config_velo_micro_bias_lines(
    reconstructed_objects,
    prefilter_regular,
    prefilter_closing,
    enable_tupling,
    preset="PbPb",
    **line_params,
):
    """
    Unified VELO micro bias lines function.

    Args:
        reconstructed_objects: Dictionary containing "velo_tracks"
        prefilter_regular: Prefilter for Hlt1VeloMicroBias line
        prefilter_closing: Prefilter for Hlt1VeloMicroBiasVeloClosing line
        preset: 'pp' or 'PbPb' configuration
        **line_params: Line-specific parameters that differ by preset:
    """

    config = VELO_MICRO_BIAS_PRESETS[preset]
    velo_tracks = reconstructed_objects["velo_tracks"]
    lines = []

    line_cfg = config["Hlt1VeloMicroBias"]
    line_cfg_closing = config["Hlt1VeloMicroBiasVeloClosing"]

    with line_maker.bind(prefilter=prefilter_regular):
        lines.append(
            line_maker(
                make_velo_micro_bias_line(
                    velo_tracks,
                    name="Hlt1VeloMicroBias",
                    enable_tupling=enable_tupling,
                    **line_cfg,
                )
            )
        )

    with line_maker.bind(prefilter=prefilter_closing):
        lines.append(
            line_maker(
                make_velo_micro_bias_line(
                    velo_tracks,
                    name="Hlt1VeloMicroBiasVeloClosing",
                    enable_tupling=enable_tupling,
                    **line_cfg_closing,
                )
            )
        )

    return lines


def config_data_quality_node(reconstructed_objects, filter_manager, **kwargs):
    """Create data quality node (special handling)."""
    reconstructed_objects_forward = hlt1_reconstruction(
        "ODQV_forward",
        with_calo=kwargs.get("with_calo", True),
        with_ut=kwargs.get("with_ut", True),
        with_muon=kwargs.get("with_muon", True),
        tracking_type=TrackingType.FORWARD,
    )

    node = make_dq_node(
        reconstructed_objects,
        reconstructed_objects_forward,
        prefilters=filter_manager.get_prefilter_set("bx"),
    )
    return node


def setup_technical_lines(
    reconstructed_objects: Dict[str, Any],
    filter_manager,
    chi2_cuts,
    preset: str = "pp",
    with_lumi: bool = True,
    enableBGI: bool = True,
    enableAlignment: bool = True,
    EnableGEC: bool = True,
    DisableLinesDuringVPClosing: bool = True,
    with_muon: bool = True,
    with_calo: bool = True,
    reco_particles: bool = True,
    enableTupling: bool = False,
    **kwargs,
) -> List[Tuple]:
    # ------------------------------------------------------------------
    # Resolve preset configuration ONCE
    # ------------------------------------------------------------------
    try:
        cfg = MONITORING_CONFIG_PRESETS[preset]
    except KeyError:
        raise ValueError(f"Unknown preset: {preset}")

    alignment_cfg = ALIGNMENT_CONFIG_PRESETS[cfg["preset_name"]]  # noqa: F841

    technical_lines = []

    # ------------------------------------------------------------------
    # Common inputs
    # ------------------------------------------------------------------
    lumiline_name = kwargs.get("lumiline_name", "Hlt1ODINLumi")
    lumilinefull_name = kwargs.get("lumilinefull_name", "Hlt1ODIN1kHzLumi")

    odin_prefilter = (
        filter_manager.get_prefilter_set("odin_only")
        if hasattr(filter_manager, "get_prefilter_set")
        else []
    )

    # ------------------------------------------------------------------
    # ODIN monitoring
    # ------------------------------------------------------------------
    technical_lines += config_odin_monitoring_lines(
        with_lumi=with_lumi,
        with_ee_far=cfg["odin"]["with_ee_far"],
        with_gec=EnableGEC and cfg["odin"]["with_gec"],
        lumiline_name=lumiline_name,
        lumilinefull_name=lumilinefull_name,
        odin_err_filter=odin_prefilter,
        velo_closed_filter=(filter_manager.get_prefilter_set("velo_state")),
        enable_tupling=enableTupling,
    )

    # ------------------------------------------------------------------
    # Error bank line
    # ------------------------------------------------------------------
    with line_maker.bind(prefilter=[sd_error_filter()]):
        technical_lines += [
            line_maker(
                make_passthrough_line(
                    name="Hlt1ErrorBank",
                    pre_scaler=cfg["error_bank_prescaler"],
                    enable_tupling=enableTupling,
                )
            )
        ]

    # ------------------------------------------------------------------
    # Minimal activity (preset-driven, explicit)
    # ------------------------------------------------------------------
    if cfg["enable_minimal_activity"]:
        with line_maker.bind(
            prefilter=filter_manager.get_prefilter_set("Hlt1MinimalActivity")
        ):
            technical_lines += [
                line_maker(
                    make_passthrough_line(
                        name="Hlt1MinimalActivity",
                        pre_scaler=cfg["minimal_activity_prescaler"],
                    )
                )
            ]

    # ------------------------------------------------------------------
    # BGI monitoring
    # ------------------------------------------------------------------
    if enableBGI:
        bgi_prefilter = filter_manager.get_prefilter_set("bgi")

        technical_lines += default_bgi_activity_lines(
            reconstructed_objects.get("pvs", []),
            reconstructed_objects.get("velo_states", []),
            prefilter=bgi_prefilter,
            enableBGI_full=cfg["enable_bgi_full"],
            enable_tupling=enableTupling,
        )

    # ------------------------------------------------------------------
    # Alignment monitoring
    # ------------------------------------------------------------------
    alignment_default_prefilters = filter_manager.get_prefilter_set(
        "alignment_default_prefilter"
    )
    if enableAlignment:
        technical_lines += config_alignment_monitoring_lines(
            reconstructed_objects,
            alignment_default_prefilters,
            chi2_cuts,
            with_muon,
            enableTupling,
            reco_particles,
            preset=cfg["preset_name"],
        )

    # ------------------------------------------------------------------
    # Velo tomography
    # ------------------------------------------------------------------
    technical_lines += config_velo_tomography_lines(
        reconstructed_objects=reconstructed_objects,
        prefilters_odin_err=odin_prefilter,
        prefilters_bx=alignment_default_prefilters,
        preset=cfg["preset_name"],
        enable_tupling=enableTupling,
    )

    # ------------------------------------------------------------------
    # Velo large clusters
    # ------------------------------------------------------------------
    technical_lines += config_velo_large_clusters_lines(
        reconstructed_objects=reconstructed_objects,
        prefilters=filter_manager.get_prefilter_set("bgi"),
        preset=cfg["preset_name"],
        enable_tupling=enableTupling,
    )

    # ------------------------------------------------------------------
    # Velo micro-bias
    # ------------------------------------------------------------------
    technical_lines += config_velo_micro_bias_lines(
        reconstructed_objects,
        prefilter_regular=filter_manager.get_prefilter_set("veloMicroBias"),
        prefilter_closing=filter_manager.get_prefilter_set("veloMicroBias_open"),
        enable_tupling=enableTupling,
        preset=cfg["preset_name"],
    )

    # ------------------------------------------------------------------
    # Beam–gas
    # ------------------------------------------------------------------
    bx_BE = filter_manager.create_filter("bx", bx_type=1)[0]  # noqa: F841

    with line_maker.bind(prefilter=filter_manager.get_prefilter_set("beam_gas")):
        technical_lines += [
            line_maker(
                make_beam_gas_line(
                    reconstructed_objects.get("velo_tracks", []),
                    reconstructed_objects.get("velo_states", []),
                    beam_crossing_type=1,
                    name="Hlt1BeamGas",
                    enable_tupling=enableTupling,
                )
            )
        ]

    return technical_lines


def add_optional_features(
    hlt1_node,
    hlt1_config,
    reconstructed_objects,
    filter_manager,
    with_fullKF: bool = False,
    with_lumi: bool = True,
    with_rich: bool = False,
    with_muon: bool = True,
    enableRateValidator: bool = True,
    data_quality: bool = False,
    **kwargs,
):
    """Add optional features to the HLT1 node."""

    if data_quality:
        return config_data_quality_node(reconstructed_objects, filter_manager, **kwargs)

    if with_lumi:
        lumiline_name = kwargs.get("lumiline_name", "Hlt1ODINLumi")
        lumilinefull_name = kwargs.get("lumilinefull_name", "Hlt1ODIN1kHzLumi")

        gather_selections = hlt1_config.get("gather_selections")

        lumi_reco = lumi_reconstruction(
            gather_selections=gather_selections,
            lumiline_name=lumiline_name,
            lumilinefull_name=lumilinefull_name,
            with_muon=with_muon,
            velo_open=kwargs.get("velo_open", False),
        )

        lumi_node = CompositeNode(
            "AllenLumiNode",
            lumi_reco["algorithms"],
            NodeLogic.NONLAZY_AND,
            force_order=False,
        )

        lumi_prefilter = (
            filter_manager.get_prefilter_set("lumi")
            if hasattr(filter_manager, "get_prefilter_set")
            else []
        )
        lumi_with_prefilter = CompositeNode(
            "LumiWithPrefilter",
            lumi_prefilter + [lumi_node],
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

    if with_fullKF and with_rich:
        hlt1_node = CompositeNode(
            "AllenWithRich",
            [hlt1_node, reconstructed_objects["rich_pid"].producer],
            NodeLogic.NONLAZY_AND,
            force_order=False,
        )

    if enableRateValidator:
        hlt1_node = CompositeNode(
            "AllenRateValidation",
            [
                hlt1_node,
                rate_validation(
                    lines=hlt1_config.get("line_algorithms", []),
                    groups=hlt1_config.get("grouped_line_algs"),
                ),
            ],
            NodeLogic.NONLAZY_AND,
            force_order=True,
        )

    user_hooks = kwargs.get("user_hooks", False)
    if user_hooks:
        hlt1_node = user_hooks(hlt1_node)

    return hlt1_node


def setup_hlt1_base(
    reconstructed_objects: Dict[str, Any],
    filter_manager,
    chi2_cuts,
    physics_lines: List[Tuple],
    smog2_lines: List[Tuple],
    technical_lines: List[Tuple],
    # Parameters with default values
    enabled_lines: List[str] = None,
    disabled_lines: List[str] = None,
    with_lumi: bool = True,
    with_rich: bool = False,
    enableRateValidator: bool = True,
    withMCChecking: bool = False,
    tracking_type=None,
    with_ut: bool = True,
    with_muon: bool = True,
    with_AC_split: bool = False,
    with_fullKF: bool = False,
    with_downstream_KF: bool = False,
    enableTupling: bool = False,
    data_quality=False,
    enableBGI: bool = True,
    enableAlignment: bool = True,
    preset: str = "pp",  # 'pp', 'PbPb', 'LightIon'
    # Additional configuration
    **kwargs,
) -> Dict[str, Any]:
    """
    Base HLT1 setup function - handles all common logic including monitoring lines.
    """
    if enabled_lines is None:
        enabled_lines = [r".*?"]
    if disabled_lines is None:
        disabled_lines = []

    hlt1_config = {"reconstruction": reconstructed_objects}

    technical_lines += setup_technical_lines(
        reconstructed_objects=reconstructed_objects,
        filter_manager=filter_manager,
        chi2_cuts=chi2_cuts,
        preset=preset,
        with_lumi=with_lumi,
        with_muon=with_muon,
        enableBGI=enableBGI,
        enableAlignment=enableAlignment,
        enableTupling=enableTupling,
        **kwargs,
    )

    grouped_lines = {
        "Physics": physics_lines,
        "SMOG2": smog2_lines,
        "Technical": technical_lines,
    }
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
    hlt1_config["grouped_line_algs"] = grouped_line_algs
    hlt1_config.update(persistency_algorithms)

    hlt1_node = add_optional_features(
        hlt1_node,
        hlt1_config,
        reconstructed_objects,
        filter_manager,
        with_lumi=with_lumi,
        with_rich=with_rich,
        with_fullKF=with_fullKF,
        with_ut=with_ut,
        with_muon=with_muon,
        data_quality=data_quality,
        enableRateValidator=enableRateValidator,
        **kwargs,
    )

    if not withMCChecking:
        hlt1_config["control_flow_node"] = hlt1_node
    else:
        validation_node = validator_node(
            reconstructed_objects,
            line_algorithms,
            includes_matching(tracking_type),
            with_ut,
            with_muon,
            with_rich,
            with_AC_split,
            with_fullKF,
            with_downstream_KF,
            filter_manager.get_prefilter_set("default"),
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
