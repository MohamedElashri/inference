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

from AllenCore.algorithms import (
    data_provider_t,
    rich_decode_pd_t,
    rich_global_pid_from_cones_t,
    rich_global_pid_from_reco_t,
    rich_init_pid_t,
    rich_make_hypos_t,
    rich_make_pixels_from_pd_t,
    rich_pd_to_smartid_t,
    rich_photon_predicted_pixel_signal_t,
    rich_photon_reconstruction_t,
    rich_pix2track_t,
    rich_quartic_signals_t,
    rich_raytrace_cherenkov_cones_t,
    rich_simple_pid_t,
)
from AllenCore.generator import make_algorithm

from AllenConf.rich_reco_options import default_rich_reco_options_allen
from AllenConf.utils import initialize_number_of_events

RICH_1 = 1
RICH_2 = 2

VALID_RICHS = [RICH_1, RICH_2]


def decode_rich(rich=RICH_1, options=default_rich_reco_options_allen()):
    if rich not in VALID_RICHS:
        raise ValueError(f"rich must be one of {VALID_RICHS}")

    number_of_events = initialize_number_of_events()

    rich_banks = make_algorithm(
        data_provider_t, name=f"rich{rich}_banks", bank_type=f"Rich{rich}"
    )

    rich_decoding = make_algorithm(
        rich_decode_pd_t,
        name=f"rich{rich}_decoding",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_raw_bank_version_t=rich_banks.host_raw_bank_version_t,
        dev_rich_raw_input_t=rich_banks.dev_raw_banks_t,
        dev_rich_raw_input_offsets_t=rich_banks.dev_raw_offsets_t,
        dev_rich_raw_input_sizes_t=rich_banks.dev_raw_sizes_t,
        dev_rich_raw_input_types_t=rich_banks.dev_raw_types_t,
        current_rich=rich,
    )

    rich_smartids = make_algorithm(
        rich_pd_to_smartid_t,
        name=f"rich{rich}_smartid",
        dev_rich_pd_offsets_t=rich_decoding.dev_rich_pd_offsets_t,
        host_rich_total_number_of_hits_t=rich_decoding.host_rich_total_number_of_hits_t,
        dev_pd_pixels_t=rich_decoding.dev_pd_pixels_t,
        current_rich=rich,
    )

    return {
        "dev_smart_ids": rich_smartids.dev_smart_ids_t,
        "dev_rich_hit_offsets": rich_decoding.dev_rich_hit_offsets_t,
        "host_rich_total_number_of_hits": rich_decoding.host_rich_total_number_of_hits_t,
        "dev_pd_pixels": rich_decoding.dev_pd_pixels_t,
        "dev_rich_pd_offsets": rich_decoding.dev_rich_pd_offsets_t,
    }


def make_pixels(
    decoded_rich=None, rich=RICH_1, options=default_rich_reco_options_allen()
):
    if rich not in VALID_RICHS:
        raise ValueError(f"rich must be one of {VALID_RICHS}")

    if decoded_rich is None:
        decoded_rich = decode_rich(rich, options)

    rich_pixels = make_algorithm(
        rich_make_pixels_from_pd_t,
        name=f"rich{rich}_make_pixels_from_pd",
        dev_rich_pd_offsets_t=decoded_rich["dev_rich_pd_offsets"],
        host_rich_total_number_of_hits_t=decoded_rich["host_rich_total_number_of_hits"],
        dev_pd_pixels_t=decoded_rich["dev_pd_pixels"],
        current_rich=rich,
    )

    return {
        "host_number_of_pixels": decoded_rich["host_rich_total_number_of_hits"],
        "dev_rich_pixels_lpos": rich_pixels.dev_rich_pixels_lpos_t,
        "dev_rich_pixels_gpos": rich_pixels.dev_rich_pixels_gpos_t,
        "dev_rich_pixels_smartid": decoded_rich["dev_smart_ids"],
        "dev_rich_pd_offsets": decoded_rich["dev_rich_pd_offsets"],
        "dev_rich_hit_offsets": decoded_rich["dev_rich_hit_offsets"],
        "dev_pd_pixels": decoded_rich["dev_pd_pixels"],
    }


def make_hypos(
    tracks, rich=RICH_1, track_name="Long", options=default_rich_reco_options_allen()
):
    if options["DetectableYieldsPrecision"] != "Average":
        raise ValueError(
            "Allen only support DetectableYieldsPrecision = Average for now"
        )
    if options["TkCKResTreatment"] != "Parameterised":
        raise ValueError("Allen only support TkCKResTreatment = Parameterised for now")

    hypos = make_algorithm(
        rich_make_hypos_t,
        name=f"rich{rich}_make_hypos",
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        dev_rich_entry_states_t=tracks[f"dev_kalman_R{rich}_F_view"],
        dev_rich_exit_states_t=tracks[f"dev_kalman_R{rich}_B_view"],
        dev_offsets_rich_states_t=tracks["dev_offsets_long_tracks"],
        current_rich=rich,
        # RadScale=,
        # MinRadiatorPathLength=,
        UseYieldWeightedAngles=options["UseYieldWeightedAngles"],
    )

    return {
        "dev_segs_best_point": hypos.dev_segs_best_point_t,
        "dev_segs_point_at_panel": hypos.dev_segs_point_at_panel_t,
        "dev_segs_best_momentum": hypos.dev_segs_best_momentum_t,
        "dev_rich_hypos": hypos.dev_rich_hypos_t,
    }


def make_photons(
    pixels,
    tracks,
    hypos,
    rich=RICH_1,
    track_name="Long",
    options=default_rich_reco_options_allen(),
):
    if rich not in VALID_RICHS:
        raise ValueError(f"rich must be one of {VALID_RICHS}")

    number_of_events = initialize_number_of_events()

    photonSel = options["PhotonSelection"]
    nSigmaCuts = options["nSigmaCuts"][photonSel][options["TkCKResTreatment"]][
        track_name
    ]

    rich_photons = make_algorithm(
        rich_photon_reconstruction_t,
        name=f"rich{rich}_photon_reconstruction",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        dev_rich_pixels_lpos_t=pixels["dev_rich_pixels_lpos"],
        dev_rich_pixels_gpos_t=pixels["dev_rich_pixels_gpos"],
        dev_rich_pd_offsets_t=pixels["dev_rich_hit_offsets"],
        dev_offsets_rich_states_t=tracks["dev_offsets_long_tracks"],
        dev_segs_point_at_panel_t=hypos["dev_segs_point_at_panel"],
        dev_segs_best_point_t=hypos["dev_segs_best_point"],
        dev_segs_best_momentum_t=hypos["dev_segs_best_momentum"],
        dev_rich_hypos_t=hypos["dev_rich_hypos"],
        current_rich=rich,
        # RadScale=,
        PreSelMinTrackROI=options["MinMaxTrackROICuts"][photonSel]["Min"],
        PreSelMaxTrackROI=options["MinMaxTrackROICuts"][photonSel]["Max"],
        # ScaleFactorCKTheta=,
        # ScaleFactorSepG=,
        PreSelNSigma=nSigmaCuts[0],
        # CKThetaBiasCorr=,
        MinAllowedCherenkovTheta=options["MinMaxCKThetaCuts"][photonSel]["Min"],
        MaxAllowedCherenkovTheta=options["MinMaxCKThetaCuts"][photonSel]["Max"],
        NSigma=nSigmaCuts[1],
    )

    rich_cones = make_algorithm(
        rich_raytrace_cherenkov_cones_t,
        name=f"rich{rich}_raytrace_cherenkov_cones",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        dev_offsets_rich_states_t=tracks["dev_offsets_long_tracks"],
        dev_segs_best_point_t=hypos["dev_segs_best_point"],
        dev_segs_best_momentum_t=hypos["dev_segs_best_momentum"],
        dev_rich_hypos_t=hypos["dev_rich_hypos"],
        current_rich=rich,
        NRingPointsMin=options["NRayTracingRingPointsMin"],
        NRingPointsMax=options["NRayTracingRingPointsMax"],
    )

    rich_photon_signals = make_algorithm(
        rich_photon_predicted_pixel_signal_t,
        name=f"rich{rich}photon_predicted_pixel_signal",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        host_number_of_photons_t=rich_photons.host_total_number_of_photons_t,
        dev_rich_hypos_t=hypos["dev_rich_hypos"],
        dev_rich_pd_offsets_t=pixels["dev_rich_pd_offsets"],
        dev_offsets_rich_photons_t=rich_photons.dev_offsets_rich_photons_t,
        dev_rich_photons_t=rich_photons.dev_rich_photons_t,
        current_rich=rich,
        MinPhotonProbability=options["MinPhotonProbabilityCuts"][photonSel],
    )

    return {
        "host_number_of_photons": rich_photons.host_total_number_of_photons_t,
        "dev_rich_photons": rich_photons.dev_rich_photons_t,
        "dev_offsets_rich_photons": rich_photons.dev_offsets_rich_photons_t,
        "dev_photon_pix_signals": rich_photon_signals.dev_photon_pix_signals_t,
        "host_total_number_of_geomeffs": rich_cones.host_total_number_of_geomeffs_t,
        "dev_rich_geomeff_offsets": rich_cones.dev_rich_geomeff_offsets_t,
        "dev_rich_geomeff_pd_ids": rich_cones.dev_rich_geomeff_pd_ids_t,
        "dev_rich_geomeff_fractions": rich_cones.dev_rich_geomeff_fractions_t,
        "dev_rich_geomeff_fractions_per_hypo": rich_cones.dev_rich_geomeff_fractions_per_hypo_t,
        "dev_track_total_signals": rich_cones.dev_track_total_signals_t,
    }


def make_signals(
    tracks,
    hypos,
    pixels,
    rich=RICH_1,
    track_name="Long",
    options=default_rich_reco_options_allen(),
):
    number_of_events = initialize_number_of_events()

    photonSel = options["PhotonSelection"]
    nSigmaCuts = options["nSigmaCuts"][photonSel][options["TkCKResTreatment"]][
        track_name
    ]

    # default for Parametrized / Long is (6, 14), but this assume we don't
    # take into account the PD size, which we do, so reduce the value for rich2
    nSigmaCuts = [(6, 7)]

    signals = make_algorithm(
        rich_quartic_signals_t,
        name=f"rich{rich}_quartic_signals",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        dev_offsets_rich_states_t=tracks["dev_offsets_long_tracks"],
        dev_segs_best_point_t=hypos["dev_segs_best_point"],
        dev_segs_point_at_panel_t=hypos["dev_segs_point_at_panel"],
        dev_segs_best_momentum_t=hypos["dev_segs_best_momentum"],
        dev_rich_hypos_t=hypos["dev_rich_hypos"],
        dev_rich_pd_offsets_t=pixels["dev_rich_pd_offsets"],
        dev_pd_pixels_t=pixels["dev_pd_pixels"],
        current_rich=rich,
        PreSelMinTrackROI=options["MinMaxTrackROICuts"][photonSel]["Min"],
        PreSelMaxTrackROI=options["MinMaxTrackROICuts"][photonSel]["Max"],
        PreSelNSigma=nSigmaCuts[0],
    )

    rich_photon_signals = make_algorithm(
        rich_photon_predicted_pixel_signal_t,
        name=f"rich{rich}photon_predicted_pixel_signal_2",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        host_number_of_photons_t=signals.host_total_number_of_photons_t,
        dev_rich_hypos_t=hypos["dev_rich_hypos"],
        dev_rich_pd_offsets_t=pixels["dev_rich_pd_offsets"],
        dev_offsets_rich_photons_t=signals.dev_offsets_rich_photons_t,
        dev_rich_photons_t=signals.dev_rich_photons_t,
        current_rich=rich,
        MinPhotonProbability=options["MinPhotonProbabilityCuts"][photonSel],
    )

    return {
        # Expected signals:
        "host_total_number_of_geomeffs": signals.host_total_number_of_geomeffs_t,
        "dev_rich_geomeff_offsets": signals.dev_rich_geomeff_offsets_t,
        "dev_rich_geomeff_pd_ids": signals.dev_rich_geomeff_pd_ids_t,
        "dev_rich_geomeff_pd_fractions": signals.dev_rich_geomeff_pd_fractions_t,
        "dev_rich_geomeff_fractions": signals.dev_rich_geomeff_fractions_t,
        # Observed signals:
        "host_number_of_photons": signals.host_total_number_of_photons_t,
        "dev_offsets_rich_photons": signals.dev_offsets_rich_photons_t,
        "dev_rich_photons": signals.dev_rich_photons_t,
        "dev_photon_pix_signals": rich_photon_signals.dev_photon_pix_signals_t,  # signals.dev_photon_pix_signals_t
    }


def make_pix2track(pixels, tracks, photons, rich=RICH_1):
    pix2track = make_algorithm(
        rich_pix2track_t,
        name=f"rich{rich}_build_pix2track",
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        host_number_of_pixels_t=pixels["host_number_of_pixels"],
        host_number_of_photons_t=photons["host_number_of_photons"],
        dev_offsets_rich_photons_t=photons["dev_offsets_rich_photons"],
        dev_rich_photons_t=photons["dev_rich_photons"],
    )

    return {
        "dev_pix2track_offsets": pix2track.dev_pix2track_offsets_t,
        "dev_pix2track": pix2track.dev_pix2track_t,
        "dev_pix2photon": pix2track.dev_pix2photon_t,
    }


def make_simple_pid(
    tracks, photons_r1, photons_r2, options=default_rich_reco_options_allen()
):
    rich_pid = make_algorithm(
        rich_simple_pid_t,
        name="rich_simple_pid",
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        dev_track_total_signals_r1_t=photons_r1["dev_track_total_signals"],
        dev_track_total_signals_r2_t=photons_r2["dev_track_total_signals"],
    )

    return {"dev_pid": rich_pid.dev_pid_t}


def init_pid(tracks, options=default_rich_reco_options_allen()):
    rich_pid = make_algorithm(
        rich_init_pid_t,
        name="rich_init_pid",
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
    )

    return {"dev_pid": rich_pid.dev_pid_t}


def make_global_pid(
    pixels, tracks, photons, pid, pix2track, options=default_rich_reco_options_allen()
):
    number_of_events = initialize_number_of_events()

    background_method = options["BackgroundEstimationMethod"]
    if background_method not in ("FromReco", "FromCones"):
        raise ValueError(
            f"BackgroundEstimationMethod must be one of ('FromReco', 'FromCones'), got {background_method!r}"
        )
    if background_method == "FromCones":
        # Rec/HLT2 like path
        rich_global_pid_t = rich_global_pid_from_cones_t
        geomeff_kwargs = dict(
            dev_rich_geomeff_offsets_r1_t=photons[RICH_1]["dev_rich_geomeff_offsets"],
            dev_rich_geomeff_pd_ids_r1_t=photons[RICH_1]["dev_rich_geomeff_pd_ids"],
            dev_rich_geomeff_fractions_r1_t=photons[RICH_1][
                "dev_rich_geomeff_fractions"
            ],
            dev_rich_geomeff_fractions_per_hypo_r1_t=photons[RICH_1][
                "dev_rich_geomeff_fractions_per_hypo"
            ],
            dev_rich_hypos_r1_t=photons[RICH_1]["dev_rich_hypos"],
            dev_rich_geomeff_offsets_r2_t=photons[RICH_2]["dev_rich_geomeff_offsets"],
            dev_rich_geomeff_pd_ids_r2_t=photons[RICH_2]["dev_rich_geomeff_pd_ids"],
            dev_rich_geomeff_fractions_r2_t=photons[RICH_2][
                "dev_rich_geomeff_fractions"
            ],
            dev_rich_geomeff_fractions_per_hypo_r2_t=photons[RICH_2][
                "dev_rich_geomeff_fractions_per_hypo"
            ],
            dev_rich_hypos_r2_t=photons[RICH_2]["dev_rich_hypos"],
        )
    else:
        # Alternative quartic reco "FromReco" path
        rich_global_pid_t = rich_global_pid_from_reco_t
        geomeff_kwargs = dict(
            dev_rich_geomeff_offsets_r1_t=photons[RICH_1]["dev_rich_geomeff_offsets"],
            dev_rich_geomeff_pd_ids_r1_t=photons[RICH_1]["dev_rich_geomeff_pd_ids"],
            dev_rich_geomeff_pd_fractions_r1_t=photons[RICH_1][
                "dev_rich_geomeff_pd_fractions"
            ],
            dev_rich_geomeff_offsets_r2_t=photons[RICH_2]["dev_rich_geomeff_offsets"],
            dev_rich_geomeff_pd_ids_r2_t=photons[RICH_2]["dev_rich_geomeff_pd_ids"],
            dev_rich_geomeff_pd_fractions_r2_t=photons[RICH_2][
                "dev_rich_geomeff_pd_fractions"
            ],
        )
    rich_global_pid = make_algorithm(
        rich_global_pid_t,
        name="rich_global_pid",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_tracks_t=tracks["host_number_of_reconstructed_scifi_tracks"],
        dev_offsets_tracks_t=tracks["dev_offsets_long_tracks"],
        dev_pid_in_t=pid["dev_pid"],
        host_number_of_pixels_r1_t=pixels[RICH_1]["host_number_of_pixels"],
        host_number_of_photons_r1_t=photons[RICH_1]["host_number_of_photons"],
        dev_rich_pd_offsets_r1_t=pixels[RICH_1]["dev_rich_pd_offsets"],
        dev_offsets_rich_photons_r1_t=photons[RICH_1]["dev_offsets_rich_photons"],
        dev_rich_photons_r1_t=photons[RICH_1]["dev_rich_photons"],
        dev_photon_pix_signals_r1_t=photons[RICH_1]["dev_photon_pix_signals"],
        dev_track_total_signals_r1_t=photons[RICH_1]["dev_track_total_signals"],
        host_number_of_pixels_r2_t=pixels[RICH_2]["host_number_of_pixels"],
        host_number_of_photons_r2_t=photons[RICH_2]["host_number_of_photons"],
        dev_rich_pd_offsets_r2_t=pixels[RICH_2]["dev_rich_pd_offsets"],
        dev_offsets_rich_photons_r2_t=photons[RICH_2]["dev_offsets_rich_photons"],
        dev_rich_photons_r2_t=photons[RICH_2]["dev_rich_photons"],
        dev_photon_pix_signals_r2_t=photons[RICH_2]["dev_photon_pix_signals"],
        dev_track_total_signals_r2_t=photons[RICH_2]["dev_track_total_signals"],
        # pix2track + pix2photon
        dev_pix2track_offsets_r1_t=pix2track[RICH_1]["dev_pix2track_offsets"],
        dev_pix2track_r1_t=pix2track[RICH_1]["dev_pix2track"],
        dev_pix2photon_r1_t=pix2track[RICH_1]["dev_pix2photon"],
        dev_pix2track_offsets_r2_t=pix2track[RICH_2]["dev_pix2track_offsets"],
        dev_pix2track_r2_t=pix2track[RICH_2]["dev_pix2track"],
        dev_pix2photon_r2_t=pix2track[RICH_2]["dev_pix2photon"],
        # alg settings
        nLikelihoodIterations=options["nLikelihoodIterations"],
        IgnoreExpectedSignals=options["PDBackIgnoreExpSignals"],
        PDBckWeights=[v for pair in options["PDBckWeights"] for v in pair],
        PDBackThresholds=[v for pair in options["PDBackThresholds"] for v in pair],
        PDBackMinPixBackground=[
            v for pair in options["PDBackMinPixBackground"] for v in pair
        ],
        PDBackMaxPixBackground=[
            v for pair in options["PDBackMaxPixBackground"] for v in pair
        ],
        **geomeff_kwargs,
    )

    return {
        "dev_pid": rich_global_pid.dev_pid_out_t,
        "dev_dll": rich_global_pid.dev_dll_out_t,
        "dev_bkgs": [
            rich_global_pid.dev_pix_bkg_r1_t,
            rich_global_pid.dev_pix_bkg_r2_t,
        ],
    }


def make_rich(track_name, tracks, options=default_rich_reco_options_allen()):
    background_method = options["BackgroundEstimationMethod"]
    if background_method not in ("FromReco", "FromCones"):
        raise ValueError(
            f"BackgroundEstimationMethod must be one of ('FromReco', 'FromCones'), got {background_method!r}"
        )

    pixels = {}
    hypos = {}
    photons = {}
    pix2tracks = {}
    for rich in VALID_RICHS:
        pixels[rich] = make_pixels(rich=rich, options=options)

        hypos[rich] = make_hypos(
            tracks, rich=rich, track_name=track_name, options=options
        )

        if background_method == "FromCones":
            # HLT2-like photon and signal reco
            photons[rich] = make_photons(
                pixels[rich],
                tracks,
                hypos[rich],
                rich=rich,
                track_name=track_name,
                options=options,
            )
        else:
            # FromReco: quartic-signals alternative reco
            photons[rich] = make_signals(
                tracks,
                hypos[rich],
                pixels[rich],
                rich=rich,
                track_name=track_name,
                options=options,
            )

            photons[rich]["dev_track_total_signals"] = photons[rich][
                "dev_rich_geomeff_fractions"
            ]

        photons[rich]["dev_rich_hypos"] = hypos[rich]["dev_rich_hypos"]

        pix2tracks[rich] = make_pix2track(
            pixels[rich], tracks, photons[rich], rich=rich
        )

    pid = init_pid(tracks, options=options)
    # pid = make_simple_pid(
    #    tracks, photons[RICH_1], photons[RICH_2], options=options)

    global_pid = make_global_pid(
        pixels, tracks, photons, pid, pix2tracks, options=options
    )

    return {
        "KF_long_track": tracks,
        "pixels": pixels,
        "hypos": hypos,
        "photons": photons,
        "bkg": global_pid["dev_bkgs"],
        "pid": global_pid["dev_pid"],
        "dll": global_pid["dev_dll"],
    }
