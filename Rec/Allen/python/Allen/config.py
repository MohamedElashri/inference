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
import logging
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from importlib import import_module
from itertools import chain
from typing import Iterator

from Configurables import AllenUpdater, ApplicationMgr
from DDDB.CheckDD4Hep import UseDD4Hep

log = logging.getLogger(__name__)
from GaudiConf.LbExec import (
    EventStores,
    InputProcessTypes,
    Options,
    ProcessTypes,
    TestOptionsBase,
)
from GaudiConf.LbExec import Options as DefaultOptions
from PyConf import configurable
from PyConf.Algorithms import AllenODINProducer, ProvideConstants, ProvideODIN
from PyConf.application import (
    ApplicationOptions,
    configure,
    configure_input,
    create_or_reuse_input_provider,
)
from PyConf.components import setup_component
from PyConf.control_flow import CompositeNode, NodeLogic
from PyConf.reading import get_generator_BeamParameters, tes_root, tes_root_mc
from pydantic import model_validator


class AllenOptions(Options):
    process: ProcessTypes | None = ProcessTypes.Hlt1
    events_per_slice: int = 500
    repetitions: int = 1
    device_memory_pool: int = 500
    host_memory_pool: int = 500
    event_store: EventStores = EventStores.EvtStoreSvc
    mdf_ioalg_name: str = "ProvideRawEvent"
    root_ioalg_name: str = "ProvideEventBranches"
    tck_from_odin: bool = False
    tck_repo: str = ""
    # MBM output settings
    mbm_partition: str = ""
    mbm_partition_id: int = 0
    mbm_partition_buffers: bool = True
    mbm_buffer_name: str = "Output"
    mbm_output_batch_size: int = 1000
    mbm_nthreads: int = 2
    mbm_com_method: str = "FIFO"

    @model_validator(mode="before")
    def n_event_slots_default(cls, data):
        if "n_event_slots" in data:
            raise ValueError("n_event_slots should not be set")
        n_threads = data.get("n_threads", 1)
        events_per_slice = data.get("events_per_slice", 500)
        data["n_event_slots"] = n_threads * events_per_slice
        return data

    @contextmanager
    def apply_binds(self) -> Iterator[None]:
        tes_root_mc.global_bind(dstformat=self.dstformat)
        if self.input_process:
            tes_root.global_bind(input_process=self.input_process)
            tes_root_mc.global_bind(input_process=self.input_process)

        with super().apply_binds():
            yield


class AllenTestOptions(TestOptionsBase, AllenOptions):
    pass


def find_alg(algs, name):
    for alg in algs:
        if alg.name() == name:
            return alg
    return None


def make_MultiEventScheduler(
    options, config, end_event_incident, configurable_algs, barriers, nodes
):
    from Configurables import MultiEventScheduler

    input_provider = create_or_reuse_input_provider(options)
    config.add(input_provider)

    if not options.output_file or options.output_type == "ROOT":
        # default writer that does nothing but consume the write queue to avoid blocking
        from Configurables import OutputWriter

        output_writer = OutputWriter()
    else:
        if options.output_file.startswith("tcp://"):
            from Configurables import ZMQOutputWriter

            output_writer = ZMQOutputWriter()
        elif options.output_file.startswith("mbm://"):
            from Configurables import Allen__MBMOutput as MBMOutput

            output_writer = MBMOutput()
            output_writer.BufferName = options.mbm_buffer_name
            output_writer.OutputBatchSize = options.mbm_output_batch_size
            output_writer.NThreads = options.mbm_nthreads
            output_writer.MBMComMethod = options.mbm_com_method
            output_writer.Partition = options.mbm_partition
            output_writer.PartitionID = options.mbm_partition_id
            output_writer.PartitionBuffers = options.mbm_partition_buffers
        else:
            from Configurables import FileOutputWriter

            output_writer = FileOutputWriter()
    output_writer.NStreams = options.n_threads
    output_writer.OutputConnection = options.output_file or ""

    # setup options for large event passthrough:
    dec_reporter = find_alg(configurable_algs, "dec_reporter")
    host_routingbits_writer = find_alg(configurable_algs, "host_routingbits_writer")
    host_output_handler = find_alg(configurable_algs, "host_output_handler")
    if dec_reporter is not None:
        output_writer.TCK = dec_reporter.getProp("tck")
        output_writer.TaskId = dec_reporter.getProp("task_id")
    if host_routingbits_writer is not None:
        output_writer.routingbit_map = host_routingbits_writer.getProp("routingbit_map")
    if host_output_handler is not None:
        output_writer.DoChecksum = host_output_handler.getProp("do_checksum")

    config.add(output_writer)

    if len(barriers) != 0:
        print(
            "WARNING: MultiEventScheduler does not support barrier algs, ignoring them: ",
            barriers,
        )
    if len(options.preamble_algs) != 0:
        print(
            "WARNING: MultiEventScheduler does not support preamble algs, ignoring them: ",
            options.preamble_algs,
        )
    scheduler = config.add(
        setup_component(
            MultiEventScheduler,
            CompositeCFNodes=nodes,
            DataProducers=configurable_algs,
            NStreams=options.n_threads,
            EvtsPerSlice=options.events_per_slice,
            Repetitions=options.repetitions,
            DeviceMemoryPool=options.device_memory_pool,
            HostMemoryPool=options.host_memory_pool,
            InputProvider=input_provider,
            OutputWriter=output_writer,
            TCKFromODIN=options.tck_from_odin,
            TCKRepo=options.tck_repo,
        )
    )
    return scheduler


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


@configurable
def allen_gaudi_config(sequence="hlt1_pp_default", node_name="hlt1_node"):
    """
    Provide the Allen top-level node
    """
    if isinstance(sequence, str):
        import importlib

        seq = importlib.import_module("AllenSequences.{}".format(sequence))
        return getattr(seq, node_name)
    else:
        return sequence()


def allen_provide_odin(stream=""):
    odin_provider = ProvideODIN(
        InputProvider=create_or_reuse_input_provider(None).name()
    )
    return odin_provider.ODIN


def call_allen_decision_logger(gather_selections):
    """
    Configure GaudiAllenCountAndDumpLineDecisions to count and report
    Allen line decisions.
    """
    from AllenConf import persistency
    from PyConf.Algorithms import GaudiAllenCountAndDumpLineDecisions

    line_names = [l + "Decision" for l in persistency.line_names(gather_selections)]

    return GaudiAllenCountAndDumpLineDecisions(
        allen_number_of_active_lines=gather_selections.host_number_of_active_lines_t,
        allen_names_of_active_lines=gather_selections.host_names_of_active_lines_t,
        allen_selections=gather_selections.dev_selections_t,
        allen_selections_offsets=gather_selections.dev_selections_offsets_t,
        Hlt1LineNames=line_names,
    )


def run_allen(
    options: AllenOptions,
    sequence,
    public_tools=[],
    add_decision_logger: bool = False,
    write_all_input_leaves: bool = True,
    flagging: bool = False,
):
    from PyConf.application import create_appMgr

    create_appMgr.global_bind(make_scheduler=make_MultiEventScheduler)

    config = configure_input(options)

    # force creation of input_provider
    input_provider = create_or_reuse_input_provider(options)

    if options.output_file and options.output_type == "ROOT":
        from AllenConf.persistency import call_allen_raw_reports, make_persistency

        def make_stream_writer(persistency_algorithms):
            from Moore.LbExec import ProcessTypes
            from Moore.stream_writers import stream_writer
            from Moore.streams import Stream

            srw = call_allen_raw_reports(persistency_algorithms)
            new_raw_banks = [
                srw.OutputSelView,
                srw.OutputDecView,
                srw.OutputRoutingBitsView,
            ]

            # TODO: If lumi is enabled for this sequence, add the output banks too
            # if "lumi_reconstruction" in hlt1_config and not options.simulation:
            #    lsm = call_allen_lumi_summary()
            #    new_raw_banks.append(lsm["lumi_summary"])

            # Give stream a name: 'default'
            stream = Stream(name="default", lines=[])
            return stream_writer(
                options=options,
                stream=stream,
                process=ProcessTypes.Hlt1,
                propagate_mc=options.simulation and options.input_type == "ROOT",
                analytics=False,
                hlt_raw_banks=new_raw_banks,
                routing_bits={"default": []},
                dst_data=[],
                dec_reports=None,
                write_all_input_leaves=write_all_input_leaves,
            )

        make_persistency.global_bind(output_handler_maker=make_stream_writer)

    from AllenCore.generator import allen_runtime_options

    with allen_runtime_options.bind(InputProvider=input_provider.name()):
        hlt1_config = allen_gaudi_config(sequence=sequence)

        # Flagging implementation by removal of all pre/post scales - passthrough line needs to be included!
        if options.simulation and flagging:
            from AllenConf.persistency import line_names as hlt1_line_names

            # in case passthrough line is not included, exit with a warning messages
            configured_lines = hlt1_line_names(hlt1_config["gather_selections"])
            if "Hlt1Passthrough" not in configured_lines:
                log.error(
                    "Passthrough line not found in the Allen sequence, please add it to enable flagging..."
                )
                sys.exit()
            for line_node in hlt1_config["line_nodes"]:
                for line in line_node.children:
                    if hasattr(line, "properties"):
                        line_properties = line.properties
                        if "pre_scaler" in line_properties:
                            line_properties.update({"pre_scaler": 1.0})
                        if "post_scaler" in line_properties:
                            line_properties.update({"post_scaler": 1.0})

        allen_cf = (
            hlt1_config
            if isinstance(hlt1_config, CompositeNode)
            else hlt1_config["control_flow_node"]
        )

        non_event_data_node = setup_allen_non_event_data_service()

        allen_algs = [non_event_data_node, allen_cf]

        if add_decision_logger:
            # Check if hlt1_config contains gather selections and if so add a decision logger
            gather_selections = hlt1_config.get("gather_selections", None)
            if gather_selections is not None:
                allen_algs.append(call_allen_decision_logger(gather_selections))

        allen_node = CompositeNode(
            "allen_algorithms",
            combine_logic=NodeLogic.NONLAZY_AND,
            children=allen_algs,
            force_order=True,
        )

        config.update(
            configure(
                options,
                allen_node,
                make_odin=allen_provide_odin,
                public_tools=public_tools,
            )
        )
        return config
