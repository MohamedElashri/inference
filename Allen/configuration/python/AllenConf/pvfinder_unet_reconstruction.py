###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
###############################################################################
from AllenCore.algorithms import (
    pvfinder_unet_t,
)
from AllenConf.utils import initialize_number_of_events
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable


@configurable
def make_pvfinder_unet(fc_output,
                       weight_file="/data/home/melashri/iris/inference/cnn_weights.bin",
                       dump_validation=""):
    """
    Wire PVFinderUNet directly downstream of the FC aggregation.
    The interval_features buffer [n_events, 40, 8, 100] has the same memory
    layout as the NCW tensor [N=n_events*40, C=8, W=100] so the copy step
    (pvfinder_ncw_layout) is eliminated — saving ~122 MB per slice.

    Parameters
    ----------
    fc_output : dict
        Return value of make_pvfinder_fc(), must contain:
          - "dev_pvfinder_interval_features"
          - "host_number_of_events"
    weight_file : str
        Path to cnn_weights.bin produced by convert_cnn_weights.py.

    Returns
    -------
    dict with:
      - "dev_pvfinder_kde_output"  : final KDE float array [n_events*40*100]
      - "unet_producer"            : the pvfinder_unet algorithm node
    """
    host_number_of_events = fc_output["host_number_of_events"]

    # Direct UNet forward pass — no NCW copy step needed
    pvfinder_unet = make_algorithm(
        pvfinder_unet_t,
        name="pvfinder_unet",
        host_number_of_events_t=host_number_of_events,
        dev_pvfinder_interval_features_t=fc_output["dev_pvfinder_interval_features"],
        weight_file=weight_file,
        dump_validation=dump_validation,
    )

    return {
        "dev_pvfinder_kde_output": pvfinder_unet.dev_pvfinder_kde_output_t,
        "unet_producer": pvfinder_unet,
    }
