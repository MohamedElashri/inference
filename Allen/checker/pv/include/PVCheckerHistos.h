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
#include <CheckerInvoker.h>
#include <ROOTHeaders.h>

class PVCheckerHistos {
public:
  TFile* m_file;
  std::string const m_directory;

  PVCheckerHistos(CheckerInvoker const* invoker, std::string const& root_file, std::string const& directory);

  void accumulate(
    std::span<const AllenRecPVInfo> vec_all_rec,
    std::span<const double> vec_rec_x,
    std::span<const double> vec_rec_y,
    std::span<const double> vec_rec_z,
    std::span<const double> vec_diff_x,
    std::span<const double> vec_diff_y,
    std::span<const double> vec_diff_z,
    std::span<const double> vec_err_x,
    std::span<const double> vec_err_y,
    std::span<const double> vec_err_z,
    std::span<const int> vec_n_trinmcpv,
    std::span<const int> vec_n_mcpv,
    std::span<const int> vec_mcpv_recd,
    std::span<const int> vec_recpv_fake,
    std::span<const int> vec_mcpv_mult,
    std::span<const int> vec_recpv_mult,
    std::span<const double> vec_mcpv_zpos,
    std::span<const double> vec_mc_x,
    std::span<const double> vec_mc_y,
    std::span<const double> vec_mc_z);

  void write();

private:
  std::unique_ptr<TTree> m_tree;
  std::unique_ptr<TTree> m_mctree;
  std::unique_ptr<TTree> m_allPV;

  std::unique_ptr<TH1F> eff_vs_z;
  std::unique_ptr<TH1F> eff_vs_mult;
  std::unique_ptr<TH1F> eff_matched_vs_z;
  std::unique_ptr<TH1F> eff_matched_vs_mult;
  std::unique_ptr<TH1F> eff_norm_z;
  std::unique_ptr<TH1F> eff_norm_mult;
  std::unique_ptr<TH1F> fakes_vs_mult;
  std::unique_ptr<TH1F> fakes_norm;

  double m_diff_x = 0.;
  double m_diff_y = 0.;
  double m_diff_z = 0.;
  double m_rec_x = 0.;
  double m_rec_y = 0.;
  double m_rec_z = 0.;
  double m_err_x = 0.;
  double m_err_y = 0.;
  double m_err_z = 0.;
  int m_nmcpv = 0;
  int m_ntrinmcpv = 0;

  double m_mc_x = 0.;
  double m_mc_y = 0.;
  double m_mc_z = 0.;

  double m_x = 0.;
  double m_y = 0.;
  double m_z = 0.;
  float m_errx = 0.;
  float m_erry = 0.;
  float m_errz = 0.;
  bool m_isFake = false;

  int const m_bins_norm_z = 50;
  int const m_bins_norm_mult = 25;
  int const m_bins_fake_mult = 20;
};
