/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#pragma once
#include "VertexDefinitions.cuh"

namespace Allen::DumppingSVs {

  struct LongParticle {
    // state
    float x, y, z, tx, ty, qop, ghost_prob, ip_chi2, chi2, ndof, chi2ndof;
    // PV
    float ip, pvx, pvy, pvz;
    // More
    float px, py, pz, pt, p, eta, rho;
    // Lepton ID
    float is_muon, is_electron;
  };
  struct genericSV {
    // Basics
    int nChildren, ievent;
    // Vertex fit
    float x, y, z;
    float c00, c11, c22;

    float px, py, pz;
    float recMass;
    // More
    float armenteros_x, armenteros_y;
    float quality;
    // PV
    float pvx, pvy, pvz;
    // External
    float doca[VertexFit::max_tracks_per_sv][VertexFit::max_tracks_per_sv] {};
    // More
    float bpv_ip, fd, fdchi2, dz, drho, eta, dira;
    float ctau, mcor, minip, minp, minpt, maxp, maxpt;
    // Daughters
    std::array<LongParticle, VertexFit::max_tracks_per_sv> children;
  };
} // namespace Allen::DumppingSVs
