#!/usr/bin/env python3
###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
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
import re
import sys
from pathlib import Path

from AllenCore.configuration_options import is_allen_standalone
from Configurables import ApplicationMgr
from Configurables import Gaudi__RootCnvSvc as RootCnvSvc

import zmq

is_allen_standalone.global_bind(standalone=True)

import argparse
import ctypes
import textwrap
from threading import Thread
from time import sleep

from Allen.config import (
    allen_odin,
    configured_bank_types,
    setup_allen_non_event_data_service,
)
from DDDB.CheckDD4Hep import UseDD4Hep
from GaudiKernel.Constants import ERROR
from GaudiPython.Bindings import AppMgr, gbl
from PyConf.application import (
    ApplicationOptions,
    ComponentConfig,
    configure,
    configure_geometry_and_conditions,
    setup_component,
)

# Load Allen entry point and helpers
gbl.gSystem.Load("libAllenLib")
gbl.gSystem.Load("libBinaryDumpers")
gbl.gSystem.Load("libAllenAlgorithms")
interpreter = gbl.gInterpreter

# FIXME: Once the headers are installed properly, this should not be
# necessary anymore
allen_dir = os.environ["ALLEN_PROJECT_ROOT"]
interpreter.Declare("#include <Dumpers/IUpdater.h>")
interpreter.Declare("#include <Allen/Allen.h>")
interpreter.Declare("#include <Allen/Provider.h>")
interpreter.Declare("""
#include <GaudiKernel/IService.h>
#include <Allen/InputProvider.h>
#include <zmq.hpp>
// Helper function to cast the LHCb-implementation of the Allen
// non-event data manager to its shared interface
template<typename TO>
struct cast_service { TO* operator()(IService* svc) { return dynamic_cast<TO*>(svc); } };
uintptr_t czmq_context(zmq::context_t& ctx) { return reinterpret_cast<uintptr_t>(ctx.operator void*()); }
""")

sequence_default = os.path.join(
    os.environ["ALLEN_INSTALL_DIR"], "constants", "hlt1_pp_default.json"
)


def cast_service(return_type, svc):
    return gbl.cast_service(return_type)()(svc)


# Handle commandline arguments
# argparse.RawTextHelpFormatter lets us do the newlines ourselves
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "-g",
    dest="det_folder",
    default=os.path.join(allen_dir, "input", "detector_configuration"),
)
parser.add_argument("-n", dest="n_events", default=0)
parser.add_argument("-t", dest="threads", default=1)
parser.add_argument("--params", dest="params", default="")
parser.add_argument("-r", dest="repetitions", default=1)
parser.add_argument("-m", dest="reserve", default=1000)
parser.add_argument("--host-memory", dest="host_memory", default=200)
parser.add_argument("-v", dest="verbosity", default=3)
parser.add_argument("-p", dest="print_memory", default=0)
parser.add_argument("--sequence", dest="sequence", default=sequence_default)
parser.add_argument("-s", dest="slices", default=1)

input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--mdf", dest="mdf", default="")
input_group.add_argument("--mep", dest="mep", default="")
input_group.add_argument("--test-file-db-key", dest="tdbk")
parser.add_argument(
    "--mep-mask-source-id-top-5", action="store_true", dest="mask_top5", default=False
)
parser.add_argument(
    "--reuse-meps",
    action="store_true",
    dest="reuse_meps",
    default=False,
    help="Fill all MEP buffers once and then reuse them",
)
parser.add_argument(
    "--profile",
    dest="profile",
    type=str,
    default="",
    choices=["", "CUDA"],
    help="Add profiler start and stop calls",
)
parser.add_argument("--output-file", dest="output_file", default=None)
parser.add_argument("--output-batch-size", dest="output_batch_size", default=10)
parser.add_argument("--monitoring-save-period", dest="mon_save_period", default=0)
parser.add_argument(
    "--monitoring-filename", dest="mon_filename", default="", help="Histogram filename"
)
parser.add_argument(
    "--enable-run-changes",
    dest="enable_run_changes",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--events-per-slice",
    dest="events_per_slice",
    default=1000,
    help="number of events per batch submitted to the GPU",
)
parser.add_argument("--device", dest="device", default=0, help="Device index")
parser.add_argument(
    "--runtime",
    dest="runtime",
    type=int,
    default=300,
    help="How long to run when reusing MEPs [s]",
)
parser.add_argument(
    "--tags",
    dest="tags",
    type=str,
    default="dddb-20171122,sim-20180530-vc-md100",
    help=textwrap.dedent("""\
    geometry and conditions tags to use
    format for detdesc - [detdesc:]dddb-tag,sim-tag
    format for dd4hep  - [dd4hep:]geom-tag,cond-tag
    format for both    - detdesc:dddb-tag,sim-tag|dd4hep:geom-tag,cond-tag
    """),
)
parser.add_argument(
    "--real-data", dest="simulation", action="store_false", default=True
)
parser.add_argument(
    "--enable-monitoring-printing",
    dest="enable_monitoring_printing",
    type=int,
    default=False,
    help="Enables printing monitoring information",
)
parser.add_argument(
    "--register-monitoring-counters",
    dest="register_monitoring_counters",
    type=int,
    default=True,
    help="Enables printing counters information",
)
parser.add_argument(
    "--tck-no-bindings",
    help="Avoid using python bindings to TCK utils",
    dest="bindings",
    action="store_false",
    default=True,
)
parser.add_argument(
    "--tck-from-odin",
    help="Respect the TCK requested by ODIN",
    dest="tck_from_odin",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--python-hlt1-node",
    type=str,
    help="Name of the variable that stores the configuration in the python module or file",
    default="hlt1_node",
    dest="hlt1_node",
)

args = parser.parse_args()

runtime_lib = None
if args.profile == "CUDA":
    runtime_lib = ctypes.CDLL("libcudart.so")

options = ApplicationOptions(_enabled=False)
options.input_type = "MDF"

if args.tdbk is not None:
    from PRConfig.TestFileDB import test_file_db

    # TestFileDB key supplied, use it for all the information it can supply
    options.set_conds_from_testfiledb(args.tdbk)
    args.simulation = options.simulation

    tdb_entry = test_file_db[args.tdbk]
    format = tdb_entry.qualifiers["Format"]
    if format == "MDF":
        args.mdf = ",".join(tdb_entry.filenames)
    elif format == "MEP":
        args.mep = ",".join(tdb_entry.filenames)
    else:
        raise ValueError("Only files in MDF or MEP format are supported by allen.py")
else:
    options.simulation = True if not UseDD4Hep else args.simulation
    options.data_type = "Upgrade"

    if args.tags.find("|") != -1:
        # special case that allows giving tags for both DetDesc and DD4hep
        valid_patterns = [
            re.compile(
                "^detdesc:(?:dddb-[0-9]{8},sim-[0-9]{8}-..-m[ud].+|upgrade/.+,upgrade/.+)|dd4hep:.+,.+"
            ),  # detdesc pattern first
            re.compile(
                "^dd4hep:.+,.+|detdesc:(?:dddb-[0-9]{8},sim-[0-9]{8}-..-m[ud].+|upgrade/.+,upgrade/.+)"
            ),  # dd4hep pattern first
        ]
        if not any([pattern.match(args.tags) for pattern in valid_patterns]):
            raise argparse.ArgumentTypeError("Bad tags given!")
        tags = {}
        for entry in args.tags.split("|"):
            build, t = entry.split(":")
            tags[build] = t.split(",")
        options.dddb_tag, options.conddb_tag = tags[
            "dd4hep" if UseDD4Hep else "detdesc"
        ]
    elif args.tags.find(":") != -1:
        # special case, the tags have been defined only for a certain build
        #  - check that build matches what we're using and propagate accordingly
        valid_patterns = [
            re.compile(
                "^detdesc:(?:dddb-[0-9]{8},sim-[0-9]{8}-..-m[ud].+|upgrade/.+,upgrade/.+)"
            ),  # detdesc pattern
            re.compile("^dd4hep:.+,.+"),  # dd4hep pattern
        ]
        if not any([pattern.match(args.tags) for pattern in valid_patterns]):
            raise argparse.ArgumentTypeError("Bad tags given!")
        split_tag = args.tags.split(":")
        if split_tag[0] == "detdesc":
            if UseDD4Hep:
                raise argparse.ArgumentTypeError(
                    "Tried to give detdesc tags for a dd4hep build"
                )
        elif split_tag[0] == "dd4hep":
            if not UseDD4Hep:
                raise argparse.ArgumentTypeError(
                    "Tried to give dd4hep tags for a detdesc build"
                )
        options.dddb_tag, options.conddb_tag = split_tag[1].split(",")
    else:
        options.dddb_tag, options.conddb_tag = args.tags.split(",")

if args.register_monitoring_counters and args.mon_filename:
    fn, ext = os.path.splitext(args.mon_filename)
    options.histo_file = fn + "_gaudi" + ext

online_cond_path = "/group/online/hlt/conditions.run3/lhcb-conditions-database"
if not args.simulation:
    if os.path.exists(online_cond_path):
        if UseDD4Hep:
            options.conditions_location = "file://" + online_cond_path
        else:
            from Configurables import XmlCnvSvc

            XmlCnvSvc().OutputLevel = ERROR
            options.velo_motion_system_yaml = os.path.join(
                online_cond_path + "/Conditions/VP/Motion.yml"
            )
make_odin = allen_odin
options.finalize()
config = ComponentConfig()

configure_geometry_and_conditions(ApplicationMgr(), config, options)

# Some extra stuff for timing table
extSvc = ["ToolSvc", "AuditorSvc", "ZeroMQSvc"]

# RootCnvSvc becauce it sets up a bunch of ROOT IO stuff
rootSvc = RootCnvSvc("RootCnvSvc", EnableIncident=1)
ApplicationMgr().ExtSvc += ["Gaudi::IODataManager/IODataManager", rootSvc]

# Get Allen JSON configuration
sequence = Path(os.path.expandvars(args.sequence))
sequence_json, sequence_source = ("", "")
tck_option = re.compile(r"([^:]+):(0x[a-fA-F0-9]{8})")
if m := tck_option.match(str(sequence)):
    import json

    from Allen.tck import dependencies_from_build_manifest, sequence_from_git

    repo = m.group(1)
    tck = m.group(2)
    sequence_json, tck_info = sequence_from_git(repo, tck, use_bindings=args.bindings)
    tck_deps = tck_info["metadata"]["stack"]["projects"]
    if not sequence_json or sequence_json == "null":
        print(f"Failed to obtain configuration for TCK {tck} from repository {repo}")
        sys.exit(1)
    elif (deps := dependencies_from_build_manifest()) != tck_deps:
        print(
            f"TCK {tck} is compatible with Allen release {deps}, not with {tck_deps}."
        )
        sys.exit(1)
    else:
        print(
            f"Loaded TCK {tck} with sequence type {tck_info['type']} and label {tck_info['label']}."
        )
        sequence_source = f"{repo}:{tck}"
elif sequence.suffix in (".py", ""):
    from Allen.tck import sequence_from_python
    from AllenCore.configuration_options import is_allen_standalone

    is_allen_standalone.global_bind(standalone=True)
    sequence_json = json.dumps(
        sequence_from_python(sequence, node_name=args.hlt1_node, verbose=True),
        sort_keys=True,
    )
    sequence_source = str(sequence)
elif sequence.suffix in (".json",):
    print(sequence.resolve())
    with sequence.open() as f:
        sequence_json = f.read()
    sequence_source = str(sequence)
else:
    raise ValueError(f"Unknown type of sequence specified: {str(sequence)}")

if args.mep:
    extSvc += ["AllenConfiguration", "MEPProvider"]
    from Configurables import AllenConfiguration, MEPProvider

    allen_conf = AllenConfiguration("AllenConfiguration")
    # Newlines in a string property cause issues
    allen_conf.JSON = sequence_json.replace("\n", "")
    allen_conf.OutputLevel = 3

    mep_provider = MEPProvider()
    mep_provider.NSlices = args.slices
    mep_provider.EventsPerSlice = args.events_per_slice
    mep_provider.OutputLevel = 6 - int(args.verbosity)
    # Number of MEP buffers and number of transpose/offset threads
    mep_provider.BufferConfig = (10, 8)
    mep_provider.TransposeMEPs = False
    mep_provider.Source = "Files"
    mep_provider.MaskSourceIDTop5 = args.mask_top5
    mep_dir = os.path.expandvars(args.mep)
    meps = mep_dir.split(",")
    if os.path.isdir(mep_dir):
        mep_provider.Connections = sorted(
            [
                os.path.join(mep_dir, mep_file)
                for mep_file in os.listdir(mep_dir)
                if mep_file.endswith(".mep")
            ]
        )
    elif len(meps) == 1 and not meps[0].endswith(".mep"):
        with open(meps[0]) as list_file:
            mep_provider.Connections = [
                m
                for m in filter(
                    lambda e: len(e) != 0, (mep.strip() for mep in list_file)
                )
            ]
    else:
        mep_provider.Connections = meps

    mep_provider.LoopOnMEPs = False
    mep_provider.Preload = args.reuse_meps
    # Use this property to allocate buffers to specific NUMA domains
    # to test the effects on performance or emulate specific memory
    # traffic scenarios
    # mep_provider.BufferNUMA = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    mep_provider.EvtMax = -1 if args.n_events == 0 else args.n_events

ApplicationMgr().EvtSel = "NONE"
ApplicationMgr().ExtSvc += extSvc

if not UseDD4Hep:
    from Configurables import DDDBConf

    config.add(DDDBConf(Simulation=options.simulation, DataType="Upgrade"))
    config.add(
        setup_component(
            "CondDB",
            Upgrade=True,
            Tags={
                "DDDB": options.dddb_tag,
                "SIMCOND": options.conddb_tag,
            },
        )
    )

bank_types = configured_bank_types(sequence_json)
cf_node = setup_allen_non_event_data_service(
    allen_event_loop=True, bank_types=bank_types
)
config.update(configure(options, cf_node, make_odin=make_odin))

# Start Gaudi and get the AllenUpdater service
gaudi = AppMgr()
sc = gaudi.initialize()
if not sc.isSuccess():
    sys.exit("Failed to initialize AppMgr")

zmqSvc = gaudi.service("ZeroMQSvc", interface=gbl.IZeroMQSvc)

# options map
options = gbl.std.map("std::string", "std::string")()
params = args.params if args.params != "" else os.getenv("PARAMFILESROOT")

for flag, value in [
    ("g", args.det_folder),
    ("params", params),
    ("n", args.n_events),
    ("t", args.threads),
    ("r", args.repetitions),
    ("output-file", args.output_file),
    ("output-batch-size", args.output_batch_size),
    ("m", args.reserve),
    ("v", args.verbosity),
    ("p", args.print_memory),
    ("sequence", sequence),
    ("s", args.slices),
    ("mdf", os.path.expandvars(args.mdf)),
    ("disable-run-changes", int(not args.enable_run_changes)),
    ("monitoring-save-period", args.mon_save_period),
    ("monitoring-filename", args.mon_filename),
    ("events-per-slice", args.events_per_slice),
    ("device", args.device),
    ("host-memory", args.host_memory),
    ("tck-from-odin", int(args.tck_from_odin)),
    ("enable-monitoring-printing", int(args.enable_monitoring_printing)),
    ("register-monitoring-counters", int(args.register_monitoring_counters)),
]:
    if value is not None:
        options[flag] = str(value)

svc = gaudi.service("AllenUpdater", interface=gbl.IService)
updater = cast_service(gbl.Allen.NonEventData.IUpdater, svc)

con = gbl.std.string("")

if args.mep:
    mep_provider = gaudi.service("MEPProvider", interface=gbl.IService)
    provider = cast_service(gbl.IInputProvider, mep_provider)
else:
    provider = gbl.Allen.make_provider(options, sequence_json)

output_handler = gbl.Allen.output_handler(provider, zmqSvc, options)

# run Allen
gbl.allen.__release_gil__ = 1

# get the libzmq context out of the zmqSvc and use it to instantiate
# the pyzmq Context
cpp_ctx = zmqSvc.context()
c_ctx = gbl.czmq_context(cpp_ctx)
ctx = zmq.Context.shadow(c_ctx)

ctx.linger = -1
control = ctx.socket(zmq.PAIR)

if args.reuse_meps:
    gaudi.start()
else:
    # C++ and python strings
    connection = "inproc:///py_allen_control"
    con = gbl.std.string(connection)
    control.bind(connection)


# Call the main Allen function from a thread so we can control it from
# the main thread
def allen_thread():
    if args.profile == "CUDA":
        runtime_lib.cudaProfilerStart()

    gbl.allen(
        options,
        sequence_json,
        sequence_source,
        updater,
        provider,
        output_handler,
        zmqSvc,
        con.c_str(),
    )

    if args.profile == "CUDA":
        runtime_lib.cudaProfilerStop()


allen_thread = Thread(target=allen_thread)
allen_thread.start()

if args.reuse_meps:
    print("sleeping")
    sleep(args.runtime)
    gaudi.stop()
else:
    # READY
    msg = control.recv()
    assert msg.decode() == "READY"

    # Start the fake event loop that takes care of the geometry and
    # conditions data.
    gaudi.start()

    # Start the Allen event loop
    control.send(b"START")
    msg = control.recv()
    assert msg.decode() == "RUNNING"
    sleep(5)
    r = control.poll(0)
    if r == zmq.POLLIN:
        msg = control.recv()
        if msg.decode() == "ERROR":
            print("ERROR from Allen event loop")
            allen_thread.join()
            sys.exit(1)

    control.send(b"STOP")
    msg = control.recv()
    assert msg.decode() == "READY"
    if args.profile == "CUDA":
        runtime_lib.cudaProfilerStop()
    gaudi.stop()
    gaudi.finalize()
    control.send(b"RESET")
    msg = control.recv()
    assert msg.decode() == "NOT_READY"

allen_thread.join()
