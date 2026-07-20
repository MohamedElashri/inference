###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "COPYING".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from typing import NamedTuple


class Thresholds(NamedTuple):
    TrackMVA_alpha: float = 296.0
    TrackMVA_maxGhostProb: float = 0.5
    TrackElectronMVA_alpha: float = 0.0
    TrackMuonMVA_alpha: float = 0.0
    TrackMuonMVA_maxCorrChi2: float = 1.8  # no longer used due to NN
    D2HH_track_ip: float = 0.06
    D2HH_track_pt: float = 800.0
    D2HH_ctIPScale: float = 1.0
    SingleHighPtLepton_pt: float = 6000.0
    SingleHighPtLepton_pt_noMuonID: float = 8000.0
    TwoTrackMVA_minMVA: float = 0.9569
    TwoTrackMVA_maxGhostProb: float = 0.5
    TwoTrackKs_minTrackPt_piKs: float = 470.0
    TwoTrackKs_minTrackIPChi2_piKs: float = 50.0
    TwoTrackKs_minComboPt_Ks: float = 2500.0
    TwoTrackKs_maxEta_Ks: float = 4.2
    TwoTrackKs_min_combip: float = 0.72
    DiMuonHighMass_pt: float = 300.0
    DiMuonHighMass_maxCorrChi2: float = 10.0  # no longer used
    DiMuonDisplaced_pt: float = 500.0
    DiMuonDisplaced_ipchi2: float = 5.0
    DiMuonDisplaced_maxCorrChi2: float = 10.0  # no longer used
    DiElectronDisplaced_pt: float = 500.0
    DiElectronDisplaced_ipchi2: float = 5.0
    DiPhotonHighMass_minET: float = 2500.0
    LambdaLLDetachedTrack_track_mipchi2: float = 9.0  # fixed values
    LambdaLLDetachedTrack_combination_bpvfd: float = 1.4  # fixed values
    XiOmegaLLL_track_ipchi2: float = 9.0  # fixed values
    TrackElectronMVA_NN: float = 0.47
    TrackMuonMVA_NN: float = 0.4
    DiMuonHighMass_NN: float = 0.4
    DiElectronDisplaced_NN: float = 0.47
    DiMuonDisplaced_NN: float = 0.4
    DiMuonDisplacedSoftPT_NN: float = 0.8
    DiMuonNoIP_NN: float = 0.95  # fixed in place for now
    DownstreamKsToPiPi_minMVA_detached: float = 0.55
    DownstreamLambdaToPPi_minMVA_detached: float = 0.5
    DownstreamTwoTrackKs_minPt: float = 475.0  # not used
    DiProtonHighMass_P_minPt: float = 5000.0
    DiProtonHighMass_PP_minPt: float = 6000.0
    DownstreamGammaToEE_minPt: float = 1000.0
    DownstreamTwoTrackKs_minTrackPt_piKs: float = 475.0
    DiElectronLowMassNoIP_NN: float = 0.94
    DiElectronLowMass_NN: float = 0.73
    Quirks_maxPHI: float = 0.07
    Quirks_maxPHIDF: float = 0.06
    Quirks_minStations: int = 6
    Quirks_maxR: float = 5.0
    Quirks_hit_thresholds: int = 200
    Quirks_max_opposite_considered: int = 6
