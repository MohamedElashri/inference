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
from AllenCore.algorithms import (
    pv_beamline_extrapolate_t, pv_beamline_histo_t, pv_beamline_peak_t,
    pv_beamline_calculate_denom_t, pv_beamline_multi_fitter_t,
    pv_beamline_cleanup_t,
    pvfinder_kde_peak_finder_t, pvfinder_nn_calculate_denom_t,
    pvfinder_vertex_fitter_t, pvfinder_nn_cleanup_t)
from AllenConf.velo_reconstruction import run_velo_kalman_filter
from AllenConf.utils import initialize_number_of_events
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable


@configurable
def make_pvs(velo_tracks,
             velo_open=False,
             pv_name="",
             zmin=-541.,
             zmax=307.,
             SMOG2_pp_separation=-334.,
             Nbins=3392):

    dz = 0.25
    pp_maxTrackZ0Err = 1.5
    SMOG2_maxTrackZ0Err = 10.

    if velo_open:
        pp_minNumTracksPerVertex = 3.
        maxChi2 = 25.
        maxTrackBlChi2 = 300.
    else:
        pp_minNumTracksPerVertex = 4.
        maxChi2 = 12.
        maxTrackBlChi2 = 10.

    number_of_events = initialize_number_of_events()
    host_number_of_events = number_of_events["host_number_of_events"]
    dev_number_of_events = number_of_events["dev_number_of_events"]

    host_number_of_reconstructed_velo_tracks = velo_tracks[
        "host_number_of_reconstructed_velo_tracks"]
    dev_offsets_all_velo_tracks = velo_tracks["dev_offsets_all_velo_tracks"]
    dev_offsets_velo_track_hit_number = velo_tracks[
        "dev_offsets_velo_track_hit_number"]
    dev_velo_track_hits = velo_tracks["dev_velo_track_hits"]

    velo_states = run_velo_kalman_filter(velo_tracks, pv_name)

    pv_beamline_extrapolate = make_algorithm(
        pv_beamline_extrapolate_t,
        name="pv_beamline_extrapolate" + pv_name,
        host_number_of_reconstructed_velo_tracks_t=
        host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_velo_states_view_t=velo_states[
            "dev_velo_kalman_beamline_states_view"])

    pv_beamline_histo = make_algorithm(
        pv_beamline_histo_t,
        name="pv_beamline_histo" + pv_name,
        host_number_of_events_t=host_number_of_events,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvtracks_t=pv_beamline_extrapolate.dev_pvtracks_t,
        maxTrackBlChi2=maxTrackBlChi2,
        zmin=zmin,
        zmax=zmax,
        dz=dz,
        Nbins=Nbins,
        SMOG2_pp_separation=SMOG2_pp_separation,
        SMOG2_maxTrackZ0Err=SMOG2_maxTrackZ0Err,
        pp_maxTrackZ0Err=pp_maxTrackZ0Err)

    pv_beamline_peak = make_algorithm(
        pv_beamline_peak_t,
        name="pv_beamline_peak" + pv_name,
        host_number_of_events_t=host_number_of_events,
        dev_zhisto_t=pv_beamline_histo.dev_zhisto_t,
        zmin=zmin,
        dz=dz,
        Nbins=Nbins,
        SMOG2_pp_separation=SMOG2_pp_separation,
        SMOG2_maxTrackZ0Err=SMOG2_maxTrackZ0Err,
        pp_maxTrackZ0Err=pp_maxTrackZ0Err)

    pv_beamline_calculate_denom = make_algorithm(
        pv_beamline_calculate_denom_t,
        name="pv_beamline_calculate_denom" + pv_name,
        host_number_of_reconstructed_velo_tracks_t=
        host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvtracks_t=pv_beamline_extrapolate.dev_pvtracks_t,
        dev_zpeaks_t=pv_beamline_peak.dev_zpeaks_t,
        dev_number_of_zpeaks_t=pv_beamline_peak.dev_number_of_zpeaks_t)

    pv_beamline_multi_fitter = make_algorithm(
        pv_beamline_multi_fitter_t,
        name="pv_beamline_multi_fitter" + pv_name,
        host_number_of_events_t=host_number_of_events,
        host_number_of_reconstructed_velo_tracks_t=
        host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvtracks_t=pv_beamline_extrapolate.dev_pvtracks_t,
        dev_zpeaks_t=pv_beamline_peak.dev_zpeaks_t,
        dev_number_of_zpeaks_t=pv_beamline_peak.dev_number_of_zpeaks_t,
        dev_pvtracks_denom_t=pv_beamline_calculate_denom.dev_pvtracks_denom_t,
        maxChi2=maxChi2,
        pp_minNumTracksPerVertex=pp_minNumTracksPerVertex,
        zmin=zmin,
        zmax=zmax,
        SMOG2_pp_separation=SMOG2_pp_separation)

    pv_beamline_cleanup = make_algorithm(
        pv_beamline_cleanup_t,
        name="pv_beamline_cleanup" + pv_name,
        host_number_of_events_t=host_number_of_events,
        dev_multi_fit_vertices_t=pv_beamline_multi_fitter.
        dev_multi_fit_vertices_t,
        dev_number_of_multi_fit_vertices_t=pv_beamline_multi_fitter.
        dev_number_of_multi_fit_vertices_t)

    return {
        "dev_number_of_zpeaks":
        pv_beamline_peak.dev_number_of_zpeaks_t,
        "dev_multi_final_vertices":
        pv_beamline_cleanup.dev_multi_final_vertices_t,
        "dev_number_of_multi_final_vertices":
        pv_beamline_cleanup.dev_number_of_multi_final_vertices_t,
        "pp_minNumTracksPerVertex":
        pp_minNumTracksPerVertex
    }


def pv_finder(velo_open=False):
    from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
    decoded_velo = decode_velo()
    velo_tracks = make_velo_tracks(decoded_velo)
    pvs = make_pvs(velo_tracks, velo_open)
    alg = pvs["dev_multi_final_vertices"].producer
    return alg


@configurable
def make_nn_pvs(velo_tracks,
                velo_open=False,
                pv_name="",
                weight_file="/data/home/melashri/iris/inference/cnn_weights.bin",
                dump_validation=""):
    """
    Full NN-based PV chain: VeloFeatureExtraction → FC aggregation → UNet
    → KDE peak finder → calculate denom → vertex fitter → cleanup.

    Drop-in replacement for make_pvs(): returns the same output keys so that
    downstream HLT1 consumers and the PVChecker are transparent to which chain
    produced the vertices.

    Parameters
    ----------
    velo_tracks : dict
        Return value of make_velo_tracks().
    velo_open : bool
        If True, loosen per-vertex track multiplicity cut to 3 (VELO open run).
    pv_name : str
        Suffix appended to algorithm names (allows multiple PV chains to coexist).
    weight_file : str
        Path to cnn_weights.bin for PVFinderUNet.
    dump_validation : str
        Directory for binary validation dumps; empty string disables dumping.

    Returns
    -------
    dict with keys matching make_pvs() output:
      - "dev_number_of_zpeaks"
      - "dev_multi_final_vertices"
      - "dev_number_of_multi_final_vertices"
      - "pp_minNumTracksPerVertex"
    """
    from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
    from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet

    pp_minNumTracksPerVertex = 3 if velo_open else 4

    number_of_events = initialize_number_of_events()
    host_number_of_events = number_of_events["host_number_of_events"]

    host_number_of_reconstructed_velo_tracks = velo_tracks[
        "host_number_of_reconstructed_velo_tracks"]

    # ------------------------------------------------------------------ #
    # 1. Velo Kalman filter + beamline extrapolation                       #
    #    (PVTrack objects shared with the classical chain if both run)     #
    # ------------------------------------------------------------------ #
    velo_states = run_velo_kalman_filter(velo_tracks, pv_name)

    pv_beamline_extrapolate = make_algorithm(
        pv_beamline_extrapolate_t,
        name="pv_beamline_extrapolate" + pv_name,
        host_number_of_reconstructed_velo_tracks_t=
        host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_velo_states_view_t=velo_states[
            "dev_velo_kalman_beamline_states_view"])

    # ------------------------------------------------------------------ #
    # 2. FC feature extraction + aggregation                              #
    # ------------------------------------------------------------------ #
    fc_output = make_pvfinder_fc(velo_tracks, pv_name)

    # ------------------------------------------------------------------ #
    # 3. UNet inference → KDE output [n_events * 4000]                   #
    # ------------------------------------------------------------------ #
    unet_output = make_pvfinder_unet(
        fc_output,
        weight_file=weight_file,
        dump_validation=dump_validation)

    # ------------------------------------------------------------------ #
    # 4. KDE peak finder → NN z-seeds                                    #
    # ------------------------------------------------------------------ #
    nn_kde_peak_finder = make_algorithm(
        pvfinder_kde_peak_finder_t,
        name="pvfinder_kde_peak_finder" + pv_name,
        host_number_of_events_t=host_number_of_events,
        dev_pvfinder_kde_output_t=unet_output["dev_pvfinder_kde_output"])

    # ------------------------------------------------------------------ #
    # 5. Calculate per-track denominator over NN z-seeds                 #
    # ------------------------------------------------------------------ #
    nn_calculate_denom = make_algorithm(
        pvfinder_nn_calculate_denom_t,
        name="pvfinder_nn_calculate_denom" + pv_name,
        host_number_of_reconstructed_velo_tracks_t=
        host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvtracks_t=pv_beamline_extrapolate.dev_pvtracks_t,
        dev_nn_zpeaks_t=nn_kde_peak_finder.dev_nn_zpeaks_t,
        dev_nn_number_of_zpeaks_t=nn_kde_peak_finder.dev_nn_number_of_zpeaks_t)

    # ------------------------------------------------------------------ #
    # 6. Adaptive vertex fitter (Tukey-bisquare, pp-only)                #
    # ------------------------------------------------------------------ #
    nn_vertex_fitter = make_algorithm(
        pvfinder_vertex_fitter_t,
        name="pvfinder_vertex_fitter" + pv_name,
        host_number_of_events_t=host_number_of_events,
        host_number_of_reconstructed_velo_tracks_t=
        host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvtracks_t=pv_beamline_extrapolate.dev_pvtracks_t,
        dev_nn_pvtracks_denom_t=nn_calculate_denom.dev_nn_pvtracks_denom_t,
        dev_nn_zpeaks_t=nn_kde_peak_finder.dev_nn_zpeaks_t,
        dev_nn_number_of_zpeaks_t=nn_kde_peak_finder.dev_nn_number_of_zpeaks_t,
        pp_minNumTracksPerVertex=pp_minNumTracksPerVertex)

    # ------------------------------------------------------------------ #
    # 7. Cleanup: deduplicate + z-sort → final PV::Vertex array          #
    #    Parameter tag names match pv_beamline_cleanup_t so that the     #
    #    PVChecker and GaudiAllenPVsToPrimaryVertexContainer are          #
    #    transparent to which chain produced the vertices.                #
    # ------------------------------------------------------------------ #
    nn_cleanup = make_algorithm(
        pvfinder_nn_cleanup_t,
        name="pvfinder_nn_cleanup" + pv_name,
        host_number_of_events_t=host_number_of_events,
        dev_multi_fit_vertices_t=
        nn_vertex_fitter.dev_nn_multi_fit_vertices_t,
        dev_number_of_multi_fit_vertices_t=
        nn_vertex_fitter.dev_nn_number_of_multi_fit_vertices_t)

    return {
        "dev_number_of_zpeaks":
        nn_kde_peak_finder.dev_nn_number_of_zpeaks_t,
        "dev_multi_final_vertices":
        nn_cleanup.dev_multi_final_vertices_t,
        "dev_number_of_multi_final_vertices":
        nn_cleanup.dev_number_of_multi_final_vertices_t,
        "pp_minNumTracksPerVertex":
        pp_minNumTracksPerVertex
    }
