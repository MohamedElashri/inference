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


def allen_specific_default_rich_reco_options(init_override_opts={}):
    opts = {"UseYieldWeightedAngles": True, "BackgroundEstimationMethod": "FromReco"}

    opts.update(init_override_opts)

    return opts


def default_rich_reco_options_allen(init_override_opts={}):
    return default_rich_reco_options(
        allen_specific_default_rich_reco_options(init_override_opts)
    )


# The config bellow is copied from Moore's rich_reconstruction.py and is meant to be
# used only in Allen standalone, in a stack setup, uses the defaults from Moore

GeV = 1000


# Returns max CK theta resolution values by track type
def defaultMaxCKThetaResolutions():
    return {
        "MC": (0.0030, 0.0015),
        "Long": (0.0030, 0.0015),
        "Down": (0.0050, 0.002),
        "Up": (0.0080, 0.002),
        "Seed": (0.0050, 0.005),
    }


# Returns the default min/max allowed cherenov theta cuts
def defaultMinMaxCKThetaCuts():
    return {
        "Nominal": {"Min": (0.005, 0.005), "Max": (0.055, 0.032)},
        "Online": {"Min": (0.005, 0.005), "Max": (0.080, 0.035)},
        "None": {"Min": (0.005, 0.005), "Max": (0.150, 0.120)},
    }


# Returns the default Track ROI pre-sel cuts
def defaultMinMaxTrackROICuts():
    return {
        "Nominal": {"Min": (0, 0), "Max": (110, 165)},
        "Online": {"Min": (0, 0), "Max": (120, 175)},
        "None": {"Min": (0, 0), "Max": (200, 200)},
    }


# Returns min CK photon probability values
def defaultMinPhotonProbabilityCuts():
    return {"Nominal": (1e-15, 1e-15), "Online": (1e-35, 1e-35), "None": (0, 0)}


# Returns the default NSigma Cuts Dictionary
def defaultNSigmaCuts():
    return {
        "Nominal": {
            "Functional": {
                "Down": [(6, 22), (3.6, 4.5)],
                "Long": [(5, 24), (3.6, 4.5)],
                "Seed": [(5, 5), (3.6, 4.5)],
                "Up": [(7, 7), (3.6, 4.5)],
                "MC": [(5, 24), (3.6, 4.5)],
            },
            "Parameterised": {
                "Down": [(6.5, 14), (3.5, 4.0)],
                "Long": [(6, 14), (4.0, 4.5)],
                "Seed": [(5, 5), (3.5, 3.5)],
                "Up": [(6, 6), (3.5, 3.5)],
                "MC": [(6, 14), (4.0, 4.5)],
            },
        },
        "Online": {
            "Functional": {
                "Down": [(45, 35), (30, 12)],
                "Long": [(45, 35), (30, 12)],
                "Seed": [(45, 35), (30, 12)],
                "Up": [(45, 35), (30, 12)],
                "MC": [(45, 35), (30, 12)],
            },
            "Parameterised": {
                "Down": [(35, 35), (25, 15)],
                "Long": [(35, 35), (25, 15)],
                "Seed": [(35, 35), (25, 15)],
                "Up": [(35, 35), (25, 15)],
                "MC": [(35, 35), (25, 15)],
            },
        },
        "None": {
            "Functional": {
                "Down": [(99999, 99999), (99999, 99999)],
                "Long": [(99999, 99999), (99999, 99999)],
                "Seed": [(99999, 99999), (99999, 99999)],
                "Up": [(99999, 99999), (99999, 99999)],
                "MC": [(99999, 99999), (99999, 99999)],
            },
            "Parameterised": {
                "Down": [(99999, 99999), (99999, 99999)],
                "Long": [(99999, 99999), (99999, 99999)],
                "Seed": [(99999, 99999), (99999, 99999)],
                "Up": [(99999, 99999), (99999, 99999)],
                "MC": [(99999, 99999), (99999, 99999)],
            },
        },
    }


def default_rich_reco_options(init_override_opts={}):
    """
    Returns a dict of the default RICH reconstruction options

    Args:
        init_override_opts (dict): override of the initial opts.
    """

    # General options. Those most likely users will want to tweak..
    opts = {
        # The mass hypotheses to consider
        "Particles": [
            "electron",
            "muon",
            "pion",
            "kaon",
            "proton",
            "deuteron",
            "belowThreshold",
        ],
        # The RICH radiators to use
        "RichGases": ["Rich1Gas", "Rich2Gas"],
        # enable 4D reco in each radiator
        "Enable4DReco": (False, False),
        # Maximum number of clusters GEC cut
        "MaxPixelClusters": 200000,
        # Maximum number of tracks GEC cut
        "MaxTracks": 10000,
        # PID Version
        "PIDVersion": 2,
        # Data Type
        "DataSetType": "Run3",
    }

    # More technical options
    opts.update(
        {
            # ===========================================================
            # Pixel treatment options
            # ===========================================================
            # Should pixel clustering be run, for either ( RICH1, RICH2 )
            "ApplyPixelClustering": (False, False),
            # Activate pixels in specific panels in each RICH only.
            "ActivatePanel": (True, True),
            # Average hit time expected in each RICH (for 4D reco)
            "averageHitTime": (13.03, 52.94),
            # Course time window for hit selection (in ns) (for 4D reco)
            "pixelTimeWindow": (1.0, 1.0),
            # Inner/Outer region override options
            "pixOverrideRegions": (False, False),
            "InnerRegionsX": (250, 999999),
            "InnerRegionsY": (300, 300),
            # ===========================================================
            # Track treatment options
            # ===========================================================
            # Use UT ?
            "UseUT": True,
            # Track Extrapolator type
            # "TrackExtrapolator": TrackSTEPExtrapolator, # not defined or used in Allen
            # Tay tracing ring points (Max)
            "NRayTracingRingPointsMax": (96, 96),
            # Tay tracing ring points (Min)
            "NRayTracingRingPointsMin": (16, 16),
            # Tolerence for creating new ray traced CK rings (as fraction of sat. CK theta)
            "RayTracingNewCKRingTol": (0.05, 0.1),
            # Detectable Yields Treatment
            "DetectableYieldsPrecision": "Average",
            # Treatment of track CK resolutions
            "TkCKResTreatment": "Parameterised",
            # Minimum momentum cuts
            "MinP": (0 * GeV, 0 * GeV),
            # Maximum momentum cuts
            "MaxP": (9e90 * GeV, 9e90 * GeV),
            # Minimum transverse momentum cut
            "MinPt": 0 * GeV,
            # Track resolution scale factors
            "TkCKResScaleFactors": (1.0, 1.0),
            # ===========================================================
            # Photon treatment options
            # ===========================================================
            # Photon selection cut setting
            #   Nominal - Normal tuned settings
            #   Online  - Loosen settings for Online running (wider side-bands)
            #   None    - Effectively no (pre)selection cuts.
            "PhotonSelection": "Nominal",
            # Truncate Cherenkov angles to a fixed precision
            "TruncateCKAngles": (True, True),
            # N Sigma cuts by selection, CK resolution model and track type.
            "nSigmaCuts": defaultNSigmaCuts(),
            # Min/Max photon CK theta angle cuts
            "MinMaxCKThetaCuts": defaultMinMaxCKThetaCuts(),
            # Min/Max Tack ROI pre-sel cuts
            "MinMaxTrackROICuts": defaultMinMaxTrackROICuts(),
            # Min Photon Probability cuts
            "MinPhotonProbabilityCuts": defaultMinPhotonProbabilityCuts(),
            "AssumeFlatSecondaryMirrors": (False, False),
            # Timing window for photon reconstruction (in ns) for (inner,outer) regions
            # Only used if 4D reconstruction is active
            "photonTimeWindow": ((0.25, 0.25), (0.25, 0.25)),
            # ===========================================================
            # Settings for the global PID minimisation
            # ===========================================================
            # Number of iterations of the global PID background and
            # likelihood minimisation.
            "nLikelihoodIterations": 2,
            # The following are technical settings per iteration.
            # Do not change unless you know what you are doing ;)
            # Array size must be at least as big as the number of iterations
            # Pixel background options
            # Ignore the expected signals based on the track information
            "PDBackIgnoreExpSignals": [True, False, False, False],
            # Ignore the hit data when computing backgrounds
            "PDBackIgnoreHitData": [False, False, False, False],
            # Minimum allowed pixel background value (RICH1,RICH2)
            "PDBackMinPixBackground": [(0, 0), (0, 0), (0, 0), (0, 0)],
            # Maximum allowed pixel background value (RICH1,RICH2)
            "PDBackMaxPixBackground": [(999, 999), (999, 999), (999, 999), (999, 999)],
            # Threshold values for background values
            "PDBackThresholds": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            # Background weights for each RICH
            "PDBckWeights": [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            # Likelihood minimizer options
            # Freeze out DLL values
            "TrackFreezeOutDLL": [2, 4, 5, 6],
            # Run a final check on the DLL values
            "FinalDLLCheck": [False, True, True, True],
            # Force change DLL values
            "TrackForceChangeDLL": [-1, -2, -3, -4],
            # likelihood threshold
            "LikelihoodThreshold": [-1e-2, -1e-3, -1e-4, -1e-5],
            # Maximum tracks that can change per iteration
            "MaxTrackChangesPerIt": [5, 5, 4, 3],
            # The minimum DLL signal value
            "MinSignalForNoLLCalc": [1e-3, 1e-3, 1e-3, 1e-3],
            # ===========================================================
            # Additional flags will be turned on from elsewhere
            # in case of RICH alignment monitoring.
            # ===========================================================
            # Reject ambiguous photons.
            "RejectAmbiguousPhotons": (False, False),
            # Save optional mirror data.
            "SaveMirrorData": False,
        }
    )

    # Update default options with those defined in steering script
    opts.update(init_override_opts)

    return opts
