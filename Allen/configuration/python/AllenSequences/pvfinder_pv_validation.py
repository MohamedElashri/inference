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
# Primary vertices from the beamline PV finder and from PVFinder, on the same
# events and VELO tracks, written with MC truth by pvfinder_pv_dump.
# Offline compare_pvs.py applies the former PrimaryVertexChecker matching
# rules to pvs_beamline.bin and pvs_pvfinder.bin.
from AllenConf.primary_vertex_reconstruction import make_pvs
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
from AllenConf.pvfinder_pv_reconstruction import make_pvfinder_pvs
from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenCore.algorithms import host_data_provider_t, pvfinder_pv_dump_t
from AllenCore.generator import generate, make_algorithm
from PyConf.control_flow import CompositeNode, NodeLogic


def pv_dump(pvs, name, output_filename):
    mc_banks = make_algorithm(host_data_provider_t, name="host_mc_pv_banks", bank_type="PVs")
    return make_algorithm(
        pvfinder_pv_dump_t,
        name=name,
        host_mc_pv_banks_t=mc_banks.host_raw_banks_t,
        host_mc_pv_offsets_t=mc_banks.host_raw_offsets_t,
        host_mc_pv_sizes_t=mc_banks.host_raw_sizes_t,
        dev_multi_final_vertices_t=pvs["dev_multi_final_vertices"],
        dev_number_of_multi_final_vertices_t=pvs["dev_number_of_multi_final_vertices"],
        output_filename=output_filename,
    )


velo_tracks = make_velo_tracks(decode_velo())
beamline_pvs = make_pvs(velo_tracks)
pvfinder_kde = make_pvfinder_unet(make_pvfinder_fc(velo_tracks))[
    "dev_pvfinder_kde_output"
]
pvfinder_pvs = make_pvfinder_pvs(velo_tracks, pvfinder_kde)

generate(
    CompositeNode(
        "PVFinderPVValidation",
        [
            pv_dump(beamline_pvs, "pv_dump_beamline", "pvs_beamline.bin"),
            pv_dump(pvfinder_pvs, "pv_dump_pvfinder", "pvs_pvfinder.bin"),
        ],
        NodeLogic.NONLAZY_AND,
        force_order=True,
    )
)
