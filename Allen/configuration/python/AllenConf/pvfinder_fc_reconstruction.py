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
import os

from AllenCore.algorithms import pvfinder_fc_aggregation_t
from AllenCore.generator import make_algorithm
from PyConf.control_flow import CompositeNode, NodeLogic
from PyConf.tonic import configurable

from AllenConf.filters import make_gec
from AllenConf.odin import make_bxtype, make_event_type, odin_error_filter
from AllenConf.utils import initialize_number_of_events
from AllenConf.velo_reconstruction import run_velo_kalman_filter


def pvfinder_node(producer):
    """PVFinder behind the HLT1 physics prefilters.

    The same filters, built with the same arguments, as setup_hlt1_node's
    physics lines (ODIN errors, beam-beam crossings, VELO closed, GEC), so they
    are the same algorithm instances. PVFinder then runs on the events whose
    VELO tracks HLT1 reconstructs for its lines, and the shared VELO chain keeps
    the event lists it has in hlt1_pp_default.
    """
    prefilters = [
        odin_error_filter("odin_error_filter"),
        make_bxtype(bx_type=3),
        make_event_type(
            name="ODIN_EvenType_VeloClosed", event_type="VeloOpen", invert=True
        ),
        make_gec(
            count_ut=False,
            count_velo=True,
            max_scifi_clusters=20000,
            max_velo_clusters=35000,
        ),
    ]
    return CompositeNode(
        "PVFinderWithPrefilter",
        prefilters + [producer],
        NodeLogic.LAZY_AND,
        force_order=True,
    )


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
            f'{filename}. Run: eval "$(make -s -C weights env MODEL=<name>)"'
        )
    path = os.path.join(os.path.abspath(weights_dir), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist; produce it with make -C weights verify MODEL=<name>"
        )
    return path


@configurable
def make_pvfinder_fc(velo_tracks, pv_name="", weight_file=None, dump_validation=""):
    """PVFinder FC aggregation (it computes the per-track features itself).

    weight_file: path to fc_weights.bin; defaults to
    $PVFINDER_WEIGHTS_DIR/fc_weights.bin (see pvfinder_weight_file).
    dump_validation: directory for weights/scripts/validate_fc.py dumps, "" = off.
    """
    if weight_file is None:
        weight_file = pvfinder_weight_file("fc_weights.bin")

    number_of_events = initialize_number_of_events()
    host_number_of_events = number_of_events["host_number_of_events"]

    host_number_of_reconstructed_velo_tracks = velo_tracks[
        "host_number_of_reconstructed_velo_tracks"
    ]

    velo_states = run_velo_kalman_filter(velo_tracks, pv_name)

    # FC aggregation: per-track features (9 per track, computed in its CSR
    # build), the FC network and the sum over each interval's tracks.
    pvfinder_fc_aggregation = make_algorithm(
        pvfinder_fc_aggregation_t,
        name="pvfinder_fc_aggregation" + pv_name,
        host_number_of_events_t=host_number_of_events,
        host_number_of_reconstructed_velo_tracks_t=host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_velo_states_view_t=velo_states["dev_velo_kalman_beamline_states_view"],
        weight_file=weight_file,
        dump_validation=dump_validation,
    )

    return {
        "dev_pvfinder_output_histogram": pvfinder_fc_aggregation.dev_pvfinder_output_histogram_t,
        "dev_pvfinder_interval_features": pvfinder_fc_aggregation.dev_pvfinder_interval_features_t,
        "host_pvfinder_unet_rows": pvfinder_fc_aggregation.host_pvfinder_unet_rows_t,
        "dev_pvfinder_slot_row": pvfinder_fc_aggregation.dev_pvfinder_slot_row_t,
        "dev_pvfinder_row_slot": pvfinder_fc_aggregation.dev_pvfinder_row_slot_t,
        "host_number_of_events": host_number_of_events,
    }
