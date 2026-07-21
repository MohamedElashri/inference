###############################################################################
# (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
import json
import os
from importlib import import_module

from Configurables import AllenUpdater, ApplicationMgr
from DDDB.CheckDD4Hep import UseDD4Hep
from GaudiConf.LbExec import Options as DefaultOptions
from PyConf import configurable
from PyConf.Algorithms import AllenODINProducer, ProvideConstants
from PyConf.application import ApplicationOptions, configure, configure_input
from PyConf.control_flow import CompositeNode, NodeLogic


def allen_odin(stream=""):
    return AllenODINProducer().ODIN


@configurable
def allen_non_event_data_config(
    dump_geometry=False, out_dir="geometry", beamline_offset=(0.0, 0.0)
):
    return dump_geometry, out_dir, beamline_offset


@configurable
def allen_json_sequence(sequence="hlt1_pp_default", json=None):
    """Provide the name of the Allen sequence and the json configuration file

    Args:
        sequence (string): name of the Allen sequence to run
        json: (string): path the JSON file to be used to configure the chosen Allen sequence. If `None`, a default file that corresponds to the sequence will be used.
    """
    if sequence is None and json is not None:
        sequence = os.path.splitext(os.path.basename(json))[0]

    if json is None:
        config_path = "${ALLEN_INSTALL_DIR}/constants"
        json_dir = os.path.join(os.path.expandvars("${ALLEN_INSTALL_DIR}"), "constants")
        available_sequences = [
            os.path.splitext(json_file)[0] for json_file in os.listdir(json_dir)
        ]
        if sequence not in available_sequences:
            raise AttributeError(
                "Sequence {} was not built in to Allen;available sequences: {}".format(
                    sequence, " ".join(available_sequences)
                )
            )
        json = os.path.join(config_path, "{}.json".format(sequence))
    elif not os.path.exists(json):
        raise OSError("JSON file does not exist")

    return (sequence, json)


def configured_bank_types(sequence_json):
    if type(sequence_json) == str:
        sequence_json = json.loads(sequence_json)
    bank_types = set()
    for t, n, c in sequence_json["sequence"]["configured_algorithms"]:
        props = sequence_json.get(n, {})
        if c == "ProviderAlgorithm" and not bool(props.get("empty", False)):
            bank_types.add(props["bank_type"])
    return bank_types


def setup_allen_non_event_data_service(allen_event_loop=False, bank_types=None):
    """Setup Allen non-event data

    An ExtSvc is added to the ApplicationMgr to provide the Allen non-event
    data (geometries etc.)
    """
    dump_geometry, out_dir, beamline_offset = allen_non_event_data_config()
    options = ApplicationOptions(_enabled=False)  # noqa: F841
    try:
        app = import_module(os.environ.get("GAUDIAPPNAME"))
    except ModuleNotFoundError:
        GaudiOptions = DefaultOptions
    else:
        GaudiOptions = getattr(app, "Options", DefaultOptions)  # noqa: F841

    appMgr = ApplicationMgr()
    if not UseDD4Hep:
        # MagneticFieldSvc is required for non-DD4hep builds
        appMgr.ExtSvc.append("MagneticFieldSvc")

    appMgr.ExtSvc.append(
        AllenUpdater(
            TriggerEventLoop=allen_event_loop,
            BeamlineOffset=beamline_offset,
            DumpToFile=dump_geometry,
            OutputDirectory=out_dir,
        )
    )

    # Detdesc need the ProvideConstants algorithm to be initialized first,
    # algorithms are initialized in alphabetical order:
    converters_node = CompositeNode(
        "allen_non_event_data",
        [ProvideConstants(name="AAAAProvideConstants")],
        combine_logic=NodeLogic.NONLAZY_OR,
        force_order=True,
    )

    return converters_node


def run_allen_reconstruction(options, make_reconstruction, public_tools=[]):
    """Configure the Allen reconstruction data flow

    Convenience function that configures all services and creates a data flow.

    Args:
        options (ApplicationOptions): holder of application options
        make_reconstruction: function returning a single CompositeNode object
        public_tools (list): list of public `Tool` instances to configure

    """
    from Allen.config import setup_allen_non_event_data_service

    config = configure_input(options)
    reconstruction = make_reconstruction()
    reco_node = (
        reconstruction if not hasattr(reconstruction, "node") else reconstruction.node
    )

    non_event_data_node = setup_allen_non_event_data_service()

    allen_node = CompositeNode(
        "allen_reconstruction",
        combine_logic=NodeLogic.NONLAZY_OR,
        children=[non_event_data_node, reco_node],
        force_order=True,
    )

    config.update(configure(options, allen_node, public_tools=public_tools))
    return config
