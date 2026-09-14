/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <thread>
#include <vector>
#include <array>
#include <deque>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <condition_variable>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <BackendCommon.h>
#include <Common.h>
#include <Logger.h>
#include <SystemOfUnits.h>
#include <sourceid.h>
#include <mdf_header.hpp>
#include <read_mdf.hpp>
#include <Event/RawBank.h>
#include <BankTypes.h>

#include "TransposeTypes.h"

/**
 * @brief      Get the (Allen) subdetector from the bank type
 *
 * @param      raw bank
 *
 * @return     Allen subdetector
 */
BankTypes sd_from_bank_type(LHCb::RawBank const* raw_bank);

/**
 * @brief      Get the (Allen) subdetector from the 5
 *             most-significant bits of a source ID
 *
 * @param      raw bank
 *
 * @return     Allen subdetector
 */
BankTypes sd_from_sourceID(LHCb::RawBank const* raw_bank);

/**
 * @brief      Check if any of the source IDs have a non-zero value
 *             in the 5 most-significant bits
 *
 * @param      span with banks in MDF layout
 *
 * @return     true if any of the sourceIDs has a non-zero value in
 *             its 5 most-significant bits
 */
bool check_sourceIDs(std::span<char const> bank_data);

/**
 * @brief      Use the bank type to source banks;
 *             for equal bank types compare the source IDs;
 *
 * @param      raw bank
 * @param      raw bank
 *
 * @return     bank type of a < bank type of b
 */
inline bool sort_by_bank_type(LHCb::RawBank const* a, LHCb::RawBank const* b)
{
  bool a_velo = a->type() == LHCb::RawBank::BankType::VP || a->type() == LHCb::RawBank::BankType::VPRetinaCluster;
  bool b_velo = b->type() == LHCb::RawBank::BankType::VP || b->type() == LHCb::RawBank::BankType::VPRetinaCluster;
  if (a_velo != b_velo) {
    return a_velo;
  }
  else {
    return (a->type() == b->type()) ? (a->sourceID() < b->sourceID()) : (a->type() < b->type());
  }
}

/**
 * @brief      Use the source IDs to sort banks
 *
 * @param      raw bank
 * @param      raw bank
 *
 * @return     sourceID of a < sourceID of b
 */
inline bool sort_by_sourceID(LHCb::RawBank const* a, LHCb::RawBank const* b)
{
  // Special case to avoid mixing VP and VPRetinateCluster banks
  if (
    (a->type() == LHCb::RawBank::BankType::VP || a->type() == LHCb::RawBank::BankType::VPRetinaCluster) &&
    (b->type() == LHCb::RawBank::BankType::VP || b->type() == LHCb::RawBank::BankType::VPRetinaCluster)) {
    return sort_by_bank_type(a, b);
  }
  else {
    return a->sourceID() < b->sourceID();
  }
}
