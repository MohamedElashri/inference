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
from AllenConf.thresholds.thresholds import Thresholds

threshold_settings = Thresholds(
    TrackMVA_alpha=-300,
    TrackMVA_maxGhostProb=0.8,
    TrackElectronMVA_alpha=1200.0,
    TrackMuonMVA_alpha=-1240.0,
    D2HH_track_ip=0.08,
    D2HH_track_pt=200.0,
    D2HH_ctIPScale=1.0,
    SingleHighPtLepton_pt=6000.0,
    SingleHighPtLepton_pt_noMuonID=8000.0,
    TwoTrackMVA_minMVA=0.957,
    TwoTrackMVA_maxGhostProb=0.8,
    TwoTrackKs_minTrackPt_piKs=200.0,
    TwoTrackKs_minTrackIPChi2_piKs=50.0,
    TwoTrackKs_minComboPt_Ks=1000.0,
    TwoTrackKs_maxEta_Ks=4.2,
    TwoTrackKs_min_combip=0.72,
    DiMuonHighMass_pt=0,
    DiMuonDisplaced_pt=0,
    DiMuonDisplaced_ipchi2=6,
    DiElectronDisplaced_pt=0,
    DiElectronDisplaced_ipchi2=3.76,
    DiPhotonHighMass_minET=2400,
    TrackElectronMVA_NN=0.6,
    TrackMuonMVA_NN=0.6,
    DiMuonHighMass_NN=0.6,
    DiElectronDisplaced_NN=0.6,
    DiMuonDisplaced_NN=0.6,
    DiMuonDisplacedSoftPT_NN=0.87,
    DownstreamKsToPiPi_minMVA_detached=0.55,
    DownstreamLambdaToPPi_minMVA_detached=0.51,
    DiProtonHighMass_P_minPt=5000,
    DiProtonHighMass_PP_minPt=6000,
    DownstreamGammaToEE_minPt=1500,
    DownstreamTwoTrackKs_minTrackPt_piKs=475,
    DiElectronLowMassNoIP_NN=0.94,
    DiElectronLowMass_NN=0.73,
)
