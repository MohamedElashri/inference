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
    pv_beamline_calculate_denom_t,
    pv_beamline_cleanup_t,
    pv_beamline_extrapolate_t,
    pv_beamline_multi_fitter_t,
    pvfinder_peak_t,
)
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable

from AllenConf.utils import initialize_number_of_events
from AllenConf.velo_reconstruction import run_velo_kalman_filter


@configurable
def make_pvfinder_pvs(
    velo_tracks,
    pvfinder_kde,
    pv_name="_pvfinder",
    zmin=-541.0,
    zmax=307.0,
    SMOG2_pp_separation=-334.0,
):
    """Primary vertices seeded by the PVFinder KDE.

    pvfinder_peak finds the z seeds in the KDE (pv-finder's peak finder); the
    beamline PV track association and fit then run on them exactly as in
    make_pvs (velo_open=False settings), sharing its VELO Kalman states and
    pv_beamline_extrapolate. PVFinder covers pp collisions in
    -100 < z < 300 mm only (no SMOG2).

    pvfinder_kde: dev_pvfinder_kde_output of make_pvfinder_unet.
    Returns the keys of make_pvs, so pv_validation and the other consumers of
    make_pvs accept it.
    """
    maxChi2 = 12.0
    pp_minNumTracksPerVertex = 4.0

    number_of_events = initialize_number_of_events()
    host_number_of_events = number_of_events["host_number_of_events"]
    host_number_of_reconstructed_velo_tracks = velo_tracks[
        "host_number_of_reconstructed_velo_tracks"
    ]

    velo_states = run_velo_kalman_filter(velo_tracks)

    # Same name and inputs as in make_pvs: one instance shared by both.
    pv_beamline_extrapolate = make_algorithm(
        pv_beamline_extrapolate_t,
        name="pv_beamline_extrapolate",
        host_number_of_reconstructed_velo_tracks_t=host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_velo_states_view_t=velo_states["dev_velo_kalman_beamline_states_view"],
    )

    pvfinder_peak = make_algorithm(
        pvfinder_peak_t,
        name="pvfinder_peak" + pv_name,
        host_number_of_events_t=host_number_of_events,
        dev_pvfinder_kde_output_t=pvfinder_kde,
    )

    pv_beamline_calculate_denom = make_algorithm(
        pv_beamline_calculate_denom_t,
        name="pv_beamline_calculate_denom" + pv_name,
        host_number_of_reconstructed_velo_tracks_t=host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvtracks_t=pv_beamline_extrapolate.dev_pvtracks_t,
        dev_zpeaks_t=pvfinder_peak.dev_zpeaks_t,
        dev_number_of_zpeaks_t=pvfinder_peak.dev_number_of_zpeaks_t,
    )

    pv_beamline_multi_fitter = make_algorithm(
        pv_beamline_multi_fitter_t,
        name="pv_beamline_multi_fitter" + pv_name,
        host_number_of_events_t=host_number_of_events,
        host_number_of_reconstructed_velo_tracks_t=host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvtracks_t=pv_beamline_extrapolate.dev_pvtracks_t,
        dev_zpeaks_t=pvfinder_peak.dev_zpeaks_t,
        dev_number_of_zpeaks_t=pvfinder_peak.dev_number_of_zpeaks_t,
        dev_pvtracks_denom_t=pv_beamline_calculate_denom.dev_pvtracks_denom_t,
        maxChi2=maxChi2,
        pp_minNumTracksPerVertex=pp_minNumTracksPerVertex,
        zmin=zmin,
        zmax=zmax,
        SMOG2_pp_separation=SMOG2_pp_separation,
    )

    pv_beamline_cleanup = make_algorithm(
        pv_beamline_cleanup_t,
        name="pv_beamline_cleanup" + pv_name,
        host_number_of_events_t=host_number_of_events,
        dev_multi_fit_vertices_t=pv_beamline_multi_fitter.dev_multi_fit_vertices_t,
        dev_number_of_multi_fit_vertices_t=pv_beamline_multi_fitter.dev_number_of_multi_fit_vertices_t,
    )

    return {
        "dev_number_of_zpeaks": pvfinder_peak.dev_number_of_zpeaks_t,
        "dev_zpeaks": pvfinder_peak.dev_zpeaks_t,
        "dev_multi_final_vertices": pv_beamline_cleanup.dev_multi_final_vertices_t,
        "dev_number_of_multi_final_vertices": pv_beamline_cleanup.dev_number_of_multi_final_vertices_t,
        "pp_minNumTracksPerVertex": pp_minNumTracksPerVertex,
    }
