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
from PyConf.control_flow import CompositeNode, NodeLogic

from AllenConf.calo_reconstruction import decode_calo
from AllenConf.enum_types import ActivityType
from AllenConf.filters import *
from AllenConf.hlt1_presets import *
from AllenConf.odin import make_bxtype, make_event_type, odin_error_filter, tae_filter


class FilterManager:
    """Manages event filters and prefilters with improved API."""

    def __init__(self, reconstructed_objects, preset=None, config_overrides=None):
        self.reconstructed_objects = reconstructed_objects
        self.presets = PREFILTER_MANAGER_PRESETS.copy()
        # Apply config overrides using update_preset
        if config_overrides:
            self.update_preset(**config_overrides)  # Changed from self.preset.update()

        # Handle preset
        if preset is None:
            self.preset = self.presets["pp_default"].copy()
        elif isinstance(preset, str):
            self.preset = PREFILTER_MANAGER_PRESETS[preset].copy()
        else:
            self.preset = preset.copy()

        self.parameters = PREFILTER_MANAGER_PRESETS["parameters"].copy()
        # Update with parameters from the specific preset if they exist
        # if 'parameters' is not empty, update self.parameters
        if self.preset["parameters"] is not None:
            # Recursively update nested dictionaries
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        update_dict(d[k], v)
                    else:
                        d[k] = v

            update_dict(self.parameters, self.preset["parameters"])
        self._filter_cache = {}
        self._prefilter_sets = {}
        self._composite_filters = {}

    def update_preset(self, **kwargs):
        """Update preset parameters with nested dict support."""

        def recursive_update(target, source):
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)
                ):
                    # Recursively update nested dict
                    recursive_update(target[key], value)
                else:
                    # Replace or add the value
                    target[key] = value

        recursive_update(self.presets, kwargs)
        # print the full preset
        return self

    def get_prefilter_set(self, name):
        """Get or create a prefilter set by name."""
        if name in self._prefilter_sets:
            return self._prefilter_sets[name]

        # Try to create from preset
        if name in self.presets.get("prefilter_sets", {}):
            return self.create_prefilter_set(name, self.presets["prefilter_sets"][name])

        raise KeyError(f"Prefilter set '{name}' not found")

    def create_named_prefilter(self, name, filter_specs, **kwargs):
        """Create and store a prefilter set with a name."""
        filters = []
        for spec in filter_specs:
            if isinstance(spec, str):
                # FIRST: Check if it's a reference to an ALREADY CREATED prefilter set
                if spec in self._prefilter_sets:
                    # Use the existing prefilter set
                    filters.extend(self._prefilter_sets[spec])

                # SECOND: Check if it's a predefined filter set in the preset
                elif spec in self.presets.get("prefilter_sets", {}):
                    # It's a reference to another filter set - recursively get/create it
                    if spec not in self._prefilter_sets:
                        # Create the referenced filter set first
                        self.create_prefilter_set(
                            spec, self.presets["prefilter_sets"][spec]
                        )
                    filters.extend(self._prefilter_sets[spec])

                else:
                    # It's a filter type - create it with optional config from kwargs
                    filter_config = kwargs.get(spec, {})
                    filters.extend(self.create_filter(spec, **filter_config))

            elif isinstance(spec, dict):
                filter_type = spec.pop("type")
                filters.extend(self.create_filter(filter_type, **spec))
            else:
                filters.append(spec)

        self._prefilter_sets[name] = filters
        return filters

    def create_composite(
        self, name, filter_names, logic=NodeLogic.LAZY_AND, force_order=False
    ):
        """Create a composite filter from named prefilter sets."""
        filters = []
        for filter_name in filter_names:
            if filter_name.startswith("+"):
                # Add individual filter
                filters.extend(self.create_filter(filter_name[1:]))
            else:
                # Add prefilter set
                filters.extend(self.get_prefilter_set(filter_name))

        composite = CompositeNode(
            f"{name}_composite", filters, logic, force_order=force_order
        )
        self._composite_filters[name] = composite
        return [composite]

    def create_conditional_prefilter(
        self, name, condition, true_filters, false_filters=None
    ):
        """Create conditional prefilter based on a condition."""
        if condition:
            return self.create_named_prefilter(name, true_filters)
        elif false_filters is not None:
            return self.create_named_prefilter(name, false_filters)
        return []

    # Keep existing methods...
    def create_filter(self, filter_type, **kwargs):
        """Generic filter creation with caching."""
        cache_key = f"{filter_type}_{str(kwargs)}"
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]

        filter_func = getattr(self, f"_create_{filter_type}", None)
        if not filter_func:
            raise ValueError(f"Unknown filter type: {filter_type}")

        result = filter_func(**kwargs)
        self._filter_cache[cache_key] = result
        return result

    def _create_odin(self):
        return [odin_error_filter("odin_error_filter")]

    def _create__sd_error_filter(self):
        return [sd_error_filter()]

    def _create_gec(self, **kwargs):
        config = self.preset.get("gec_config", {}).copy()
        config.update(kwargs)
        return [make_gec(**config)]

    def _create_gec_upc(self):
        params = self.parameters.get("gec_upc", {})
        decoded_calo = decode_calo()
        return [
            make_checkEcalEnergy(
                decoded_calo["dev_total_ecal_e"],
                name="CheckEcalEnergyUPC",
                ecalCut=params.get("gec_upc_ecal_cut", 94000),
                cutHigh=True,
            )
        ]

    def _create_gec_hadronic(self, ecal_cut=None):
        params = self.parameters.get("gec_hadronic", {})
        decoded_calo = decode_calo()
        return [
            make_checkEcalEnergy(
                decoded_calo["dev_total_ecal_e"],
                name="CheckEcalEnergyHadronic",
                ecalCut=params.get("min_ecal_hadro", 94000),
                cutHigh=False,
            )
        ]

    def _create_scifi_filter(self):
        params = self.parameters.get("scifi_filter", {})
        return [
            scifi_gec(
                "veloMicroBias_scifi_clusters_filter",
                min_clusters=params.get("min_clusters", 0),
                max_clusters=params.get("max_clusters", 7000),
            )
        ]

    def _create_velo_filter(self):
        params = self.parameters.get("velo_filter", {})
        return [
            velo_gec(
                "veloMicroBias_velo_clusters_filter",
                min_clusters=params.get("min_clusters", 200),
                max_clusters=params.get("max_clusters", 7000),
            )
        ]

    def _create_veloMicroBias_clusters_filter(self):
        params = self.parameters.get("veloMicroBias_clusters_filter", {})
        veloMicroBias_scifi_clusters_filter = [
            scifi_gec(
                "veloMicroBias_scifi_clusters_filter",
                min_clusters=params.get("scifi_min_clusters", 0),
                max_clusters=params.get("scifi_max_clusters", 7000),
            )
        ]
        veloMicroBias_velo_clusters_filter = [
            velo_gec(
                "veloMicroBias_velo_clusters_filter",
                min_clusters=params.get("velo_min_clusters", 200),
                max_clusters=params.get("velo_max_clusters", 7000),
            )
        ]
        return CompositeNode(
            "veloMicroBias_clusters_filter_node",
            veloMicroBias_scifi_clusters_filter + veloMicroBias_velo_clusters_filter,
            NodeLogic.LAZY_AND,
            force_order=False,
        )

    def _create_pv_activity_filter(self):
        params = self.parameters.get("pv_activity_filter", {})
        return make_minimal_activity_filter(
            self.reconstructed_objects,
            minimal_activity_type=ActivityType.PRIMARY_VERTICES,
            min_activity=params.get("pv_min_activity", 1),
            max_activity=params.get("pv_max_activity", 100),
        )

    def _create_velo_closing_filter(self):
        params = self.parameters.get("velo_closing_filter", {})

        return [
            make_gec(
                gec_name="closing_filter",
                count_velo=True,
                count_scifi=False,
                count_ut=False,
                min_velo_clusters=params.get("min_clusters", 200),
                max_velo_clusters=params.get("max_clusters", 30000),
            )
        ]

    def _create_bx_BB(self):
        return [make_bxtype(bx_type=3)]

    def _create_bx_BE(self):
        return [make_bxtype(bx_type=1)]

    def _create_bx(self, bx_type=None):
        bx_type = bx_type or self.preset.get("bx_type", 3)
        return [make_bxtype(bx_type=bx_type)]

    def _create_velo_closed(self):
        return [
            make_event_type(
                name="ODIN_EvenType_VeloClosed", event_type="VeloOpen", invert=True
            )
        ]

    def _create_velo_open(self):
        return [make_event_type(event_type="VeloOpen")]

    def _create_tae_activity_filter(self):
        return [
            make_tae_activity_filter(
                self.reconstructed_objects["long_tracks"],
                self.reconstructed_objects["velo_tracks"],
            )
        ]

    def _create_tae_filter(self):
        return [tae_filter()]

    def _create_activity_filter(self):
        activity_type = self.parameters.get("activity_type")
        if not isinstance(activity_type, ActivityType):
            raise ValueError(
                "activity_type must be an instance of ActivityType enum: ",
                activity_type,
            )
        if activity_type != ActivityType.VELO_CLUSTERS:
            raise ValueError(
                "_create_activity_filter: Please use pv_activity_filter instead."
            )
        params = self.parameters.get("activity_filter", {})
        min_activity = params.get("min_activity", 200)
        max_activity = params.get("max_activity", 999999999)

        return make_minimal_activity_filter(
            self.reconstructed_objects,
            activity_type,
            min_activity=min_activity,
            max_activity=max_activity,
        )

    def _create_lowmult(self, **kwargs):
        return make_lowmult(
            self.reconstructed_objects["velo_tracks"],
            self.reconstructed_objects["ecal_clusters"],
            **kwargs,
        )

    def _create_gec_photon_nvelo_upc(self):
        params = self.parameters.get("gec_photon_nvelo_upc", {})
        return [
            make_lowmult(
                self.reconstructed_objects["velo_tracks"],
                self.reconstructed_objects["ecal_clusters"],
                name="CheckPhotonUPC",
                maxTracks=params.get("max_tracks", 10),
                max_ecal_clusters=params.get("max_ecal_clusters", 10),
            )
        ]

    def create_prefilter_set(self, name, filter_sequence, **filter_kwargs):
        """Create a named prefilter set from a sequence of filter types."""
        filters = []
        for filter_spec in filter_sequence:
            if isinstance(filter_spec, str):
                # Simple filter type - ensure result is a list
                created_filters = self.create_filter(
                    filter_spec, **filter_kwargs.get(filter_spec, {})
                )

                # Ensure we have a list (create_filter might return a single object)
                if not isinstance(created_filters, (list, tuple)):
                    created_filters = [created_filters]

                filters.extend(created_filters)

            elif isinstance(filter_spec, dict):
                # Complex filter specification
                filter_type = filter_spec.pop("type")
                created_filters = self.create_filter(filter_type, **filter_spec)

                # Ensure we have a list
                if not isinstance(created_filters, (list, tuple)):
                    created_filters = [created_filters]

                filters.extend(created_filters)

            else:
                # Already a filter object
                filters.append(filter_spec)

        self._prefilter_sets[name] = filters
        return filters
