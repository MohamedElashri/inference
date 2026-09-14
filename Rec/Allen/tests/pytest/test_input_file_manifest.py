###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration             #
#                                                                             #
# This software is distributed under the terms of the Apache License           #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".                 #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities   #
# granted to it by virtue of its status as an Intergovernmental Organization    #
# or submit itself to any jurisdiction.                                        #
###############################################################################
from types import SimpleNamespace

import PyConf.application as application


def test_root_input_manifest(monkeypatch):
    """The Allen ROOT reader satisfies PyConf's existing manifest contract."""
    monkeypatch.setattr(application, "_rootIOAlgSingleton", None)
    options = SimpleNamespace(root_ioalg_name="ProvideEventBranches")
    reader = application.create_or_reuse_rootIOAlg(options)
    manifest = reader.InputFileManifestLocation
    assert manifest.producer is reader
    assert manifest.type == "LHCb::IO::InputFileManifest"


def test_raw_event_input_has_no_root_manifest():
    """MDF/MEP input continues to publish only a raw event."""
    # Build-time test collection can precede configurable generation.
    from Configurables import ProvideRawEvent

    assert set(ProvideRawEvent.getDefaultProperties()) >= {"RawEventLocation"}
    assert "InputFileManifestLocation" not in ProvideRawEvent.getDefaultProperties()
