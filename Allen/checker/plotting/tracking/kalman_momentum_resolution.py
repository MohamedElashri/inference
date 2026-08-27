#!/usr/bin/env python
###############################################################################
# (c) Copyright 2018-2021 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
# Need to do this before importing ROOT.
import argparse
import sys
from array import array

import numpy as np

parser = argparse.ArgumentParser(description="Draw Kalman Filter momentum resolution.")
parser.add_argument(
    "-p",
    "--momentum",
    action="store_true",
    dest="p",
    default=False,
    help="Use momentum instead of q/p (default false).",
)
parser.add_argument(
    "-s",
    "--stddev",
    action="store_true",
    dest="stddev",
    default=False,
    help="Use StdDev for resolution instead of Gaussian fit (default false).",
)
args = parser.parse_args()

# Import ROOT and LHCbStyle.
import ROOT

sys.path.append("../")
from common.Legend import *
from common.LHCbStyle import *

setLHCbStyle()
ROOT.gStyle.SetPadLeftMargin(0.15)

# Setup binning.
edges = np.array(
    [0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0]
)
zipped_edges = zip(edges[:-1], edges[1:])
bins = [zipped_edge for zipped_edge in zipped_edges]
centers = 0.5 * (edges[:-1] + edges[1:])
widths = 0.5 * (edges[1:] - edges[:-1])


# Get the momentum resolution in a single momentum bin.
def pResFit(tree, pvar, bin, stddev=False):
    th1 = ROOT.TH1F("h", "h", 100, -0.05, 0.05)
    tree.Draw(
        "(" + pvar + "-mcp_p)/mcp_p>>h",
        "ghost==0 && mcp_p>={} && mcp_p<{}".format(bin[0], bin[1]),
        "goff",
    )
    if stddev:
        return th1.GetStdDev(), th1.GetStdDev() / np.sqrt(2 * th1.Integral() - 2)
    else:
        f = ROOT.TF1("f", "gaus", -0.05, 0.05)
        th1.Fit(f, "R")
        return f.GetParameter(2), f.GetParError(2)


# Get the qOp resolution in a single momentum bin.
def qopResFit(tree, pvar, bin, stddev=False):
    th1 = ROOT.TH1F("h", "h", 100, -0.05, 0.05)
    tree.Draw(
        "(" + pvar + "-1./mcp_p)*mcp_p>>h",
        "ghost==0 && mcp_p>={} && mcp_p<{}".format(bin[0], bin[1]),
        "goff",
    )
    if stddev:
        return th1.GetStdDev(), th1.GetStdDev() / np.sqrt(2 * th1.Integral() - 2)
    else:
        f = ROOT.TF1("f", "gaus", -0.05, 0.05)
        th1.Fit(f, "R")
        return th1.GetFunction("f").GetParameter(2), th1.GetFunction("f").GetParError(2)


# Main execution.
inFile = ROOT.TFile("../../../output/KalmanIPCheckerOutput.root")
inTree = inFile.Get("kalman_ip_tree")
if args.p:
    firstRes = np.array(
        [pResFit(inTree, "1./first_qop", bin, args.stddev) for bin in bins]
    )
    bestRes = np.array(
        [pResFit(inTree, "1./best_qop", bin, args.stddev) for bin in bins]
    )
    yLabel = "#sigma_{#it{p}}/#it{p}"
else:
    firstRes = np.array(
        [qopResFit(inTree, "first_qop", bin, args.stddev) for bin in bins]
    )
    bestRes = np.array(
        [qopResFit(inTree, "best_qop", bin, args.stddev) for bin in bins]
    )
    yLabel = "#sigma_{#it{q/p}}/(#it{q/p})"

c1 = ROOT.TCanvas("c1", "c1")
centers = centers / 1000.0
widths = widths / 1000.0
firstG = ROOT.TGraphErrors(
    len(centers),
    array("d", centers),
    array("d", firstRes[:, 0]),
    array("d", widths),
    array("d", firstRes[:, 1]),
)
bestG = ROOT.TGraphErrors(
    len(centers),
    array("d", centers),
    array("d", bestRes[:, 0]),
    array("d", widths),
    array("d", bestRes[:, 1]),
)
firstG.SetMarkerColor(ROOT.kBlack)
firstG.SetLineColor(ROOT.kBlack)
bestG.SetMarkerColor(ROOT.kCyan + 1)
bestG.SetLineColor(ROOT.kCyan + 1)
mg = ROOT.TMultiGraph()
mg.Add(firstG)
mg.Add(bestG)
mg.Draw("ap")
mg.GetHistogram().GetXaxis().SetTitle("#it{p} [GeV/#it{c}^{2}]")
mg.GetHistogram().GetYaxis().SetTitle(yLabel)
mg.GetHistogram().GetYaxis().SetRangeUser(0, 0.015)
mg.GetYaxis().SetTitleOffset(1.1)
mg.GetHistogram().GetXaxis().SetRangeUser(edges[0] / 1000.0, edges[-1] / 1000.0)
legend = ROOT.TLegend(0.67, 0.92, 0.95, 0.72)
legend.AddEntry(firstG, "Initial")
legend.AddEntry(bestG, "Kalman")
legend.SetFillStyle(0)
legend.Draw("same")
fname = "../../../plotsfornote/kalman"
if args.p:
    fname += "_momentum"
else:
    fname += "_qop"
if args.stddev:
    fname += "_stddev"
else:
    fname += "_resolution"
fname += ".pdf"
c1.SaveAs(fname)
