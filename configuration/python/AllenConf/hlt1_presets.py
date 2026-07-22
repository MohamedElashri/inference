###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "COPYING".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from typing import Any, Callable, Dict, List, Optional, Tuple

from AllenConf.enum_types import ActivityType

# Central configuration presets for different running conditions
ALIGNMENT_CONFIG_PRESETS = {
    "pp": {
        "one_muon_prescaler": 6e-5,
        "include_jpsi_tagging": False,
        "disable_upsilon_alignment": False,
        "upsilon_alignment": {
            "minHighMassTrackPt": 1800.0,
            "minHighMassTrackP": 20000.0,
            "minMass": 8000.0,
            "maxMass": 150000.0,
            "minFdChi2": -1.0,
            "minIP": -1.0,
            "minDira": 0.9,
        },
        "line_maker_style": "bind_context",  # Use with line_maker.bind(prefilter=prefilters)
    },
    "PbPb": {
        "one_muon_prescaler": 0.001,
        "include_jpsi_tagging": True,
        "disable_upsilon_alignment": False,
        "upsilon_alignment": {
            "maxChi2Corr": 1.8,
            "minMass": 8000.0,
            "minHighMassTrackPt": 550,
        },
        "line_maker_style": "explicit_prefilter",  # Use line_maker(line, prefilter=prefilters)
    },
    "LightIon": {
        "one_muon_prescaler": 0.001,
        "include_jpsi_tagging": True,
        "disable_upsilon_alignment": False,
        "upsilon_alignment": {
            "maxChi2Corr": 1.8,
            "minMass": 8000.0,
            "minHighMassTrackPt": 550,
        },
        "line_maker_style": "explicit_prefilter",  # Use line_maker(line, prefilter=prefilters)
    },
}

MONITORING_CONFIG_PRESETS = {
    "pp": {
        "preset_name": "pp",
        "is_mini": False,
        "error_bank_prescaler": 0.0001,
        "enable_bgi_full": False,
        "odin": {
            "with_ee_far": True,
            "with_gec": False,
        },
        # Minimal activity
        "enable_minimal_activity": False,
    },
    "PbPb": {
        "preset_name": "PbPb",
        "is_mini": False,
        "error_bank_prescaler": 0.01,
        "enable_bgi_full": True,
        "odin": {
            "with_ee_far": False,
            "with_gec": True,
        },
        # Minimal activity
        "enable_minimal_activity": True,
        "minimal_activity_prescaler": 0.01,
    },
    "PbPb_mini": {
        "preset_name": "PbPb",
        "is_mini": True,
        "error_bank_prescaler": 0.01,
        "enable_bgi_full": False,
        "odin": {
            "with_ee_far": False,
            "with_gec": True,
        },
        # Minimal activity
        "enable_minimal_activity": True,
        "minimal_activity_prescaler": 0.01,
    },
    "LightIon": {
        "preset_name": "LightIon",
        "is_mini": False,
        "error_bank_prescaler": 0.01,
        "enable_bgi_full": False,
        "odin": {
            "with_ee_far": False,
            "with_gec": True,
        },
        # Minimal activity
        "enable_minimal_activity": True,
        "minimal_activity_prescaler": 0.01,
    },
    "LightIon_mini": {
        "preset_name": "LightIon",
        "is_mini": True,
        "error_bank_prescaler": 0.01,
        "enable_bgi_full": False,
        "odin": {
            "with_ee_far": False,
            "with_gec": True,
        },
        # Minimal activity
        "enable_minimal_activity": True,
        "minimal_activity_prescaler": 0.01,
    },
}

VELO_TOMOGRAPHY_CONFIG_PRESETS = {
    "pp": {
        "prescalers": {
            "downstreamz": 5e-4,  # When full_velo_tomography=False
            "dwfs": 0.01,  # When full_velo_tomography=False
            "downstreamz_full": 0.5,  # When full_velo_tomography=True
            "dwfs_full": 1.0,  # When full_velo_tomography=True
        },
        "add_velo_gec": False,  # Only if not full_velo_tomography
        "full_velo_tomography": False,
    },
    "PbPb": {
        "prescalers": {
            "downstreamz": 0.005,  # When full_velo_tomography=False
            "dwfs": 0.1,  # When full_velo_tomography=False
            "downstreamz_full": 0.5,  # When full_velo_tomography=True
            "dwfs_full": 1.0,  # When full_velo_tomography=True
        },
        "add_velo_gec": True,
        "full_velo_tomography": False,
    },
    "LightIon": {
        "prescalers": {
            "downstreamz": 0.005,  # When full_velo_tomography=False
            "dwfs": 0.01,  # When full_velo_tomography=False
            "downstreamz_full": 0.5,  # When full_velo_tomography=True
            "dwfs_full": 1.0,  # When full_velo_tomography=True
        },
        "add_velo_gec": True,
        "full_velo_tomography": False,
    },
}
VELO_LARGE_CLUSTERS_CONFIG_PRESETS = {
    "pp": {"enable": True, "min_eta": 5.0, "min_cluster_size": 4, "min_n_hits": 5},
    "PbPb": {"enable": True, "min_eta": 5.0, "min_cluster_size": 4, "min_n_hits": 5},
    "LightIon": {
        "enable": False,
        "min_eta": 5.0,
        "min_cluster_size": 4,
        "min_n_hits": 5,
    },
}

# Simple configuration presets for VELO micro bias lines
VELO_MICRO_BIAS_PRESETS = {
    "pp": {
        # pp configuration
        "Hlt1VeloMicroBias": {
            "pre_scaler": 1.0,
            "post_scaler": 1e-3,  # velo_micro_bias_post_scaler default
        },
        "Hlt1VeloMicroBiasVeloClosing": {
            "post_scaler": 3.0e-3,
        },
    },
    "PbPb": {
        "Hlt1VeloMicroBias": {
            "pre_scaler": 0.15,
            "post_scaler": 1.0,
            "min_velo_tracks": 3,
        },
        "Hlt1VeloMicroBiasVeloClosing": {
            "pre_scaler": 1.0,
            "post_scaler": 0.5,
        },
    },
    "LightIon": {
        "Hlt1VeloMicroBias": {
            "pre_scaler": 1.0,
            "post_scaler": 1.0,
            "min_velo_tracks": 3,
        },
        "Hlt1VeloMicroBiasVeloClosing": {
            "pre_scaler": 1.0,
            "post_scaler": 1.0,
        },
    },
}

BGI_ACTIVITY_PRESETS = {
    "BGIPVsCylAll": {
        "min_vtx_z": -3000.0,
        "max_vtx_z": 3000.0,
        "min_vtx_nTracks": 10,
    },
    "BGIPVsCylIR": {
        "enable": False,
        "max_vtx_z": 250.0,
    },
    "BGIPseudoPVsAll": {
        "min_state_z": -3000.0,
        "max_state_z": 3000.0,
        "min_local_nTracks": 10.0,
    },
    "BGIPseudoPVsUp": {
        "min_state_z": -3000.0,
        "max_state_z": -250.0,
        "min_local_nTracks": 10.0,
    },
    "BGIPseudoPVsDown": {
        "min_state_z": 250.0,
        "max_state_z": 3000.0,
        "min_local_nTracks": 10,
    },
    "BGIPseudoPVsIR": {
        "min_state_z": -250.0,
        "max_state_z": 250.0,
        "min_local_nTracks": 28.0,  # 10 for ion collision
    },
    "Hlt1BGIPseudoPVsBeamOne": {
        "pre_scaler": 1e-2,
        "post_scaler": 1.0,
    },
    "Hlt1BGIPseudoPVsBeamTwo": {
        "pre_scaler": 6.5e-2,
        "post_scaler": 1.0,
    },
    "Hlt1BGIPseudoPVsUpBeamBeam": {
        "pre_scaler": 1e-3,
        "post_scaler": 1.0,
    },
    "Hlt1BGIPseudoPVsDownBeamBeam": {
        "pre_scaler": 5e-2,
        "post_scaler": 1.0,
    },
    "Hlt1BGIPseudoPVsIRBeamBeam": {
        "enable": True,
        "pre_scaler": 4e-5,
        "post_scaler": 1.0,
    },
}

# prefilter presets
PREFILTER_MANAGER_PRESETS = {
    "pp_default": {
        "gec_config": {
            "count_ut": False,
            "count_velo": True,
            "max_scifi_clusters": 20000,
            "max_velo_clusters": 35000,
        },
        "min_z": -537.5,
        "max_z": -337.5,
        # Only put the parameters you want to override from the default 'parameters' here
        "parameters": {},
    },
    "pp_smog2_be": {
        "gec_config": {
            "count_ut": False,
            "count_velo": True,
            "max_scifi_clusters": 20000,
            "max_velo_clusters": 35000,
        },
        "min_z": -537.5,
        "max_z": -337.5,
        # Only put the parameters you want to override from the default 'parameters' here
        "parameters": {},
    },
    "PbPb_default": {
        "gec_config": {
            "count_ut": False,
            "count_velo": True,
            "max_scifi_clusters": 40000,
            "max_velo_clusters": 60000,
        },
        # Only put the parameters you want to override from the default 'parameters' here
        "parameters": {
            "velo_closing_filter": {
                "max_clusters": 5000,
            },
        },
    },
    "PbPb_mini": {
        "gec_config": {
            "count_ut": False,
            "count_velo": True,
            "max_scifi_clusters": 40000,
            "max_velo_clusters": 60000,
        },
        # Only put the parameters you want to override from the default 'parameters' here
        "parameters": {
            "velo_closing_filter": {
                "max_clusters": 5000,
            },
        },
    },
    "LightIon_default": {
        "gec_config": {
            "count_ut": False,
            "count_velo": True,
            "max_scifi_clusters": 30000,
            "max_velo_clusters": 60000,
        },
        # Only put the parameters you want to override from the default 'parameters' here
        "parameters": {
            "velo_closing_filter": {
                "max_clusters": 5000,
            },
        },
    },
    "LightIon_mini": {
        "gec_config": {
            "count_ut": False,
            "count_velo": True,
            "max_scifi_clusters": 30000,
            "max_velo_clusters": 60000,
        },
        # Only put the parameters you want to override from the default 'parameters' here
        "parameters": {
            "velo_closing_filter": {
                "max_clusters": 5000,
            },
        },
    },
    "parameters": {
        "scifi_filter": {
            # SciFi filter defaults
            "min_clusters": 0,
            "max_clusters": 7000,
        },
        "velo_filter": {
            # Velo filter defaults
            "min_clusters": 200,
            "max_clusters": 7000,
        },
        "veloMicroBias_clusters_filter": {
            # VeloMicroBias clusters filter defaults
            "scifi_min_clusters": 0,
            "scifi_max_clusters": 7000,
            "velo_min_clusters": 200,
            "velo_max_clusters": 7000,
        },
        "velo_closing_filter": {
            # Velo filter defaults
            "min_clusters": 200,
            "max_clusters": 30000,
        },
        "pv_activity_filter": {
            # PV activity defaults
            "pv_min_activity": 1,
            "pv_max_activity": 100,
        },
        "gec_upc": {
            # UPC defaults
            "ecal_cut": 94000,
        },
        "gec_hadronic": {
            "min_ecal_hadro": 94000,
        },
        "gec_photon_nvelo_upc": {
            # LowMultiplicity defaults
            "max_tracks": 10,
            "max_ecal_clusters": 10,
        },
        "activity_filter": {
            "min_activity": 200,
            "max_activity": 999999999,
        },
        "activity_type": ActivityType.VELO_CLUSTERS,
        "activity_type_closing": ActivityType.VELO_CLUSTERS,
    },
}
