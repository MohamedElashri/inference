Measure performance
=====================

Both the throughput and the physics performance are monitored over time automatically with the two following tools:

* |throughput_link|
* |dashboard_link|

.. |throughput_link| raw:: html

   <a href="https://lbgrafana.cern.ch/d/Qvm54N3Mz/allen-performance?orgId=1" target="_blank">Allen throughput evolution over time in grafana</a>

.. |dashboard_link| raw:: html

   <a href="https://lblhcbpr.cern.ch/dashboards/allen" target="_blank">Allen dashboard with physics performance over time</a>

.. _measuring_throughput:

Processing throughput
^^^^^^^^^^^^^^^^^^^^^

Throughput is measured **exclusively** through the MooreOnline testbench,
which runs Allen with the Multi Event Scheduler in the same configuration as
online data-taking (MEP input, MBM output).  Every merge request in Allen is
automatically benchmarked this way in the CI system on a number of different
GPUs and a CPU.  The results are published in this
|mattermost_channel_throughput|.

.. |mattermost_channel_throughput| raw:: html

   <a href="https://mattermost.web.cern.ch/lhcb/channels/allenpr-throughput" target="_blank">mattermost channel</a>

To measure throughput locally, use the MooreOnline testbench with a positive
``--measure-throughput`` value (in seconds).  For example::

  MooreOnline/run MooreOnline/MooreScripts/scripts/testbench.py \
    --working-dir=hlt1_throughput \
    MooreOnline/MooreScripts/tests/options/HLT1NGPU/Arch.xml \
    --test-file-db-key=2024_mep_292860_run_change_test \
    --hlt-type=hlt1_pp_default \
    --measure-throughput=60

The testbench starts the MBM/MEP producer tasks and the HLT1 task, waits for
the system to settle (``--delay-tp-measurement``), and then measures the
steady-state throughput from the published counters, printing
``Average total throughput: Evts/s = …``.

For other input files, choose the appropriate TestFileDB key (see
:ref:`input_files`).  Do not use the standalone ``./Allen`` executable for
throughput measurements; that path is deprecated.


Physics performance
^^^^^^^^^^^^^^^^^^^^^^

Physics quantities, such as track and vertex reconstruction efficiencies and
the momentum resolution, are determined by calling Allen from Moore, using the
same ``lbexec`` pattern as :ref:`run_allen_in_stack`.

Allen also provides a :ref:`root_service`, with which physics quantities used
in HLT1 lines can be stored in ROOT files for performance studies.

.. _moore_performance_scripts:

Scripts in Moore
-------------------
Moore provides ready-made reconstruction/checker option modules under
``Moore/Hlt/RecoConf/options/``, each with a matching YAML file.  For example,
to check the forward-tracking impact-parameter resolution with MC checking::

  lbexec Moore/Hlt/RecoConf/options/hlt1_reco_allen_IPresolution.py:main Moore/Hlt/RecoConf/options/hlt1_reco_allen_IPresolution.yaml

This calls the configured Allen algorithms, converts the reconstructed tracks
to Rec objects and runs the MC checkers for track reconstruction efficiencies.
Other examples (track resolution, muon-ID efficiency, calo resolution, …)
follow the same pattern.

HltEfficiencyChecker in DaVinci
----------------------------------------
The |davinci| repository contains the ``HltEfficiencyChecker`` tool for giving rates and
efficiencies. To get ``DaVinci``, you can use the nightlies or do ``make DaVinci`` from the top-level directory of the stack.

.. |davinci| raw:: html

   <a href="https://gitlab.cern.ch/lhcb/DaVinci" target="_blank">DaVinci</a>

To get the efficiencies of all the Allen lines, from the top-level directory do::

  ./DaVinci/run DaVinci/HltEfficiencyChecker/scripts/hlt_eff_checker.py DaVinci/HltEfficiencyChecker/options/hlt1_eff_default_retinacluster.yaml

and to get the rates::

  ./DaVinci/run DaVinci/HltEfficiencyChecker/scripts/hlt_eff_checker.py DaVinci/HltEfficiencyChecker/options/hlt1_rate_example_retinacluster.yaml

Full documentation for the ``HltEfficiencyChecker`` tool, including a walk-through example for HLT1 efficiencies with Allen, is given |hltefficiencychecker_tutorial|.

.. |hltefficiencychecker_tutorial| raw:: html

   <a href="https://lhcbdoc.web.cern.ch/lhcbdoc/moore/master/tutorials/hltefficiencychecker.html" target="_blank">in this tutorial</a>


Scripts for standalone Allen (deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   Standalone Allen is **deprecated** and will be removed in a subsequent set
   of merge requests.  The standalone physics-performance instructions below
   are kept for reference only.  Use :ref:`moore_performance_scripts` instead.

Create the directory Allen/output, then the ROOT file PrCheckerPlots.root will be saved there when running a validation sequence.

* Efficiency plots: Histograms of reconstructible and reconstructed tracks are saved in ``Allen/output/PrCheckerPlots.root``.
  Plots of efficiencies versus various kinematic variables can be created by running ``efficiency_plots.py <../../checker/plotting/tracking/efficiency_plots.py>`` in the directory
  ``checker/plotting/tracking``. The resulting ROOT file ``efficiency_plots.root`` with graphs of efficiencies is saved in the directory ``plotsfornote_root``.
* Momentum resolution plots: A 2D histogram of momentum resolution versus momentum is also stored in ``Allen/output/PrCheckerPlots.root`` for Upstream and Forward tracks.
  Velo tracks are straight lines, so no momentum resolution is calculated. Running the script ``momentum_resolution.py <../../checker/plotting/tracking/momentum_resolution.py>`` in the directory ``checker/plotting/tracking``
  will produce a plot of momentum resolution versus momentum in the ROOT file ``momentum_resolution.root`` in the directory ``plotsfornote_root``.
  In this script, the 2D histogram of momentum resolution versus momentum is projected onto the momentum resolution axis in slices of the momentum.
  The resulting 1D histograms are fitted with a Gaussian function if they have more than 100 entries. The Gaussian fit is constrained to the region [-0.05,0.05] in
  the case of Forward tracks and to [-0.5, 0.5] for Upstream tracks respectively to avoid the non-Gaussian tails.
  The mean and sigma of the Gaussian are used as value and uncertainty in the momentum resolution versus momentum plot.
  The plot is only generated if at least one momentum slice histogram has more than 100 entries.
