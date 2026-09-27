###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenCore.algorithms import (
    pvfinder_unet_t,
)
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable

from AllenConf.pvfinder_fc_reconstruction import pvfinder_weight_file
from AllenConf.utils import initialize_number_of_events


@configurable
def make_pvfinder_unet(fc_output, weight_file=None, dump_validation=""):
    """
    The UNet, downstream of the FC aggregation: its rows (one per interval with
    tracks) to the KDE.

    Parameters
    ----------
    fc_output : dict
        Return value of make_pvfinder_fc(): the rows and their layout, the
        number of events, and the precision and batch size, which the UNet
        takes from it so the two algorithms always agree.
    weight_file : str, optional
        Path to cnn_weights.bin. Defaults to $PVFINDER_WEIGHTS_DIR/cnn_weights.bin,
        produced by the repository's weights/ pipeline (see pvfinder_weight_file).
    dump_validation : str
        Directory for the validation dumps, "" = off.

    Returns
    -------
    dict with:
      - "dev_pvfinder_kde_output"  : final KDE float array [n_events*40*100]
      - "unet_producer"            : the pvfinder_unet algorithm node
    """
    if weight_file is None:
        weight_file = pvfinder_weight_file("cnn_weights.bin")
    host_number_of_events = fc_output["host_number_of_events"]

    pvfinder_unet = make_algorithm(
        pvfinder_unet_t,
        name="pvfinder_unet",
        host_number_of_events_t=host_number_of_events,
        dev_pvfinder_interval_features_t=fc_output["dev_pvfinder_interval_features"],
        host_pvfinder_unet_rows_t=fc_output["host_pvfinder_unet_rows"],
        dev_pvfinder_slot_row_t=fc_output["dev_pvfinder_slot_row"],
        dev_pvfinder_row_slot_t=fc_output["dev_pvfinder_row_slot"],
        weight_file=weight_file,
        precision=fc_output["precision"],
        unet_batch_events=fc_output["unet_batch_events"],
        dump_validation=dump_validation,
    )

    return {
        "dev_pvfinder_kde_output": pvfinder_unet.dev_pvfinder_kde_output_t,
        "unet_producer": pvfinder_unet,
    }
