###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
import os

from AllenCore.algorithms import (
    pvfinder_velo_feature_extraction_t,
    pvfinder_fc_aggregation_t
)
from AllenConf.velo_reconstruction import run_velo_kalman_filter
from AllenConf.utils import initialize_number_of_events
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable

def pvfinder_weight_file(filename):
    """Absolute path of a PVFinder weight file inside $PVFINDER_WEIGHTS_DIR.

    Allen has no default location for PVFinder weights. Produce them with the
    repository's weights/ pipeline (make -C weights verify MODEL=<name>) and
    export the directory it prints (make -C weights env MODEL=<name>) before
    generating the sequence configuration. Raises rather than writing an empty
    or wrong path into the configuration.
    """
    weights_dir = os.environ.get("PVFINDER_WEIGHTS_DIR", "")
    if not weights_dir:
        raise RuntimeError(
            "PVFINDER_WEIGHTS_DIR is not set; PVFinder sequences need it to find "
            f"{filename}. Run: eval \"$(make -s -C weights env MODEL=<name>)\"")
    path = os.path.join(os.path.abspath(weights_dir), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist; produce it with make -C weights verify MODEL=<name>")
    return path


@configurable
def make_pvfinder_fc(velo_tracks, pv_name="", weight_file=None, dump_validation=""):
    """PVFinder feature extraction + FC aggregation.

    weight_file: path to fc_weights.bin; defaults to
    $PVFINDER_WEIGHTS_DIR/fc_weights.bin (see pvfinder_weight_file).
    dump_validation: directory for weights/scripts/validate_fc.py dumps, "" = off.
    """
    if weight_file is None:
        weight_file = pvfinder_weight_file("fc_weights.bin")

    number_of_events = initialize_number_of_events()
    host_number_of_events = number_of_events["host_number_of_events"]

    host_number_of_reconstructed_velo_tracks = velo_tracks[
        "host_number_of_reconstructed_velo_tracks"]

    velo_states = run_velo_kalman_filter(velo_tracks, pv_name)

    # 1. Feature Extraction (9 features per track)
    pvfinder_feature_extraction = make_algorithm(
        pvfinder_velo_feature_extraction_t,
        name="pvfinder_velo_feature_extraction" + pv_name,
        host_number_of_events_t=host_number_of_events,
        host_number_of_reconstructed_velo_tracks_t=host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_velo_states_view_t=velo_states["dev_velo_kalman_beamline_states_view"]
    )

    # 2. Fused FC+Aggregation - runs the full MLP per-track in registers,
    #    accumulates directly into interval features. - No global latent buffer.
    pvfinder_fc_aggregation = make_algorithm(
        pvfinder_fc_aggregation_t,
        name="pvfinder_fc_aggregation" + pv_name,
        host_number_of_events_t=host_number_of_events,
        host_number_of_reconstructed_velo_tracks_t=host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvfinder_track_features_t=pvfinder_feature_extraction.dev_pvfinder_track_features_t,
        weight_file=weight_file,
        dump_validation=dump_validation,
    )

    return {
        "dev_pvfinder_output_histogram": pvfinder_fc_aggregation.dev_pvfinder_output_histogram_t,
        "dev_pvfinder_interval_features": pvfinder_fc_aggregation.dev_pvfinder_interval_features_t,
        "host_number_of_events": host_number_of_events,
    }
