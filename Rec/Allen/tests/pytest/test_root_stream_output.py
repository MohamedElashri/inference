###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License           #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".                 #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from types import SimpleNamespace

import pytest


def test_run_allen_root_stream_output(monkeypatch):
    from Allen import config
    from GaudiConf.LbExec import ProcessTypes
    from PyConf import application

    writer = object()
    reports = SimpleNamespace(
        OutputSelView=object(), OutputDecView=object(), OutputRoutingBitsView=object()
    )
    options = SimpleNamespace(
        output_file="output.dst", output_type="ROOT", simulation=True, input_type="ROOT"
    )

    # Allen does not depend on Moore: model the stream-writer boundary here.
    def stream_writer(*, output_location, **kwargs):
        assert output_location.location == "/Event/DAQ/RawEvent"
        assert kwargs["process"] is ProcessTypes.Hlt1
        assert kwargs["hlt_raw_banks"] == [
            reports.OutputSelView,
            reports.OutputDecView,
            reports.OutputRoutingBitsView,
        ]
        assert kwargs["propagate_mc"]
        return writer

    class CallbackChecked(Exception):
        pass

    def check_callback(*, output_handler_maker):
        # make_persistency expands this iterable into its control-flow children.
        assert [*output_handler_maker({})] == [writer]
        raise CallbackChecked

    monkeypatch.setitem(config.sys.modules, "Moore", SimpleNamespace())
    monkeypatch.setitem(
        config.sys.modules, "Moore.LbExec", SimpleNamespace(ProcessTypes=ProcessTypes)
    )
    monkeypatch.setitem(
        config.sys.modules, "Moore.streams", SimpleNamespace(Stream=SimpleNamespace)
    )
    monkeypatch.setitem(
        config.sys.modules,
        "Moore.stream_writers",
        SimpleNamespace(
            stream_writer=stream_writer,
            RootOutputLocation=lambda location: SimpleNamespace(location=location),
        ),
    )
    monkeypatch.setitem(
        config.sys.modules,
        "AllenConf.persistency",
        SimpleNamespace(
            call_allen_raw_reports=lambda _: reports,
            make_persistency=SimpleNamespace(global_bind=check_callback),
        ),
    )
    monkeypatch.setattr(
        application, "create_appMgr", SimpleNamespace(global_bind=lambda **_: None)
    )
    monkeypatch.setattr(config, "configure_input", lambda _: {})
    monkeypatch.setattr(config, "create_or_reuse_input_provider", lambda _: None)

    with pytest.raises(CallbackChecked):
        config.run_allen(options, sequence="unused")
