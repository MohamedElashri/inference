###############################################################################
# (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file " COPYING ".            #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenConf.thresholds.thresholds import Thresholds

threshold_settings = Thresholds(
    TrackMVA_alpha=40,
    TrackMVA_maxGhostProb=0.8,
    TrackElectronMVA_alpha=1470,
    TrackMuonMVA_alpha=-1060,
    D2HH_track_ip=0.09,
    D2HH_track_pt=700,
    D2HH_ctIPScale=1.0,
    SingleHighPtLepton_pt=12500,
    SingleHighPtLepton_pt_noMuonID=12500,
    TwoTrackMVA_minMVA=0.9695,
    TwoTrackMVA_maxGhostProb=0.8,
    TwoTrackKs_minTrackPt_piKs=762.889,
    TwoTrackKs_minTrackIPChi2_piKs=76.6865,
    TwoTrackKs_minComboPt_Ks=2500,
    TwoTrackKs_maxEta_Ks=4.2,
    TwoTrackKs_min_combip=1.38,
    DiMuonHighMass_pt=850,
    DiMuonDisplaced_pt=200,
    DiMuonDisplaced_ipchi2=6.32,
    DiElectronDisplaced_pt=425,
    DiElectronDisplaced_ipchi2=7.76,
    DiPhotonHighMass_minET=2800,
    TrackElectronMVA_NN=0.7,
    TrackMuonMVA_NN=0.6,
    DiMuonHighMass_NN=0.6,
    DiElectronDisplaced_NN=0.7,
    DiMuonDisplaced_NN=0.6,
    DiMuonDisplacedSoftPT_NN=0.87,
    DownstreamKsToPiPi_minMVA_detached=0.55,
    DownstreamLambdaToPPi_minMVA_detached=0.55,
    DiProtonHighMass_P_minPt=5000,
    DiProtonHighMass_PP_minPt=6000,
    DownstreamGammaToEE_minPt=1500,
    DownstreamTwoTrackKs_minTrackPt_piKs=550,
    DiElectronLowMassNoIP_NN=0.94,
    DiElectronLowMass_NN=0.73,
)
