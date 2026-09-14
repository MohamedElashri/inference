Run Allen
==========

Allen is run natively on GPU through the Gaudi **Multi Event Scheduler**
(:ref:`multi_event_scheduler`).  The recommended ways to run Allen are:

* **within the LHCb stack, steered by Moore** (offline / integration), using
  ``lbexec`` — described below;
* **online / data-taking**, through the MooreOnline testbench and
  ``AllenConfig.py`` — described in :ref:`run_allen_online`.

.. _run_allen_in_stack:

Running Allen within the stack (Gaudi/Moore)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Allen is configured as the HLT1 application inside Moore.  The option scripts
follow the standard Moore ``lbexec`` pattern: a Python module defines a
``main(options)`` function, and the run-time options are passed as a YAML
file.

From the top-level stack directory::

  lbexec Moore/Hlt/Hlt1Conf/options/allen_hlt1_pp_default.py:main Moore/Hlt/Hlt1Conf/options/allen_hlt1_pp_default.yaml

The option module
(`allen_hlt1_pp_default.py <https://gitlab.cern.ch/lhcb/Moore/-/blob/master/Hlt/Hlt1Conf/options/allen_hlt1_pp_default.py>`_)
is simply::

  from Allen.config import AllenTestOptions, run_allen

  def main(options: AllenTestOptions):
      return run_allen(options, sequence="hlt1_pp_default")

and the YAML file
(`allen_hlt1_pp_default.yaml <https://gitlab.cern.ch/lhcb/Moore/-/blob/master/Hlt/Hlt1Conf/options/allen_hlt1_pp_default.yaml>`_)
selects the input and number of events::

  testfiledb_key: "upgrade_Sept2022_minbias_0fb_md_mdf"
  dddb_tag: "upgrade/dddb-20231017-new-particle-table"
  evt_max: 1000

To run a different sequence, either change the ``sequence`` argument passed to
``run_allen`` in the option module, or use one of the ready-made modules such
as ``hlt1_allen_lowenergy.py``.  The full set of ``AllenOptions`` (``n_threads``,
``events_per_slice``, ``device_memory_pool``, ``output_file``, ``output_type``,
``tck_from_odin``, …) is defined in ``Allen.config.AllenOptions`` and can
be overridden in the YAML file.

If the option module does not already type its ``main`` with
``AllenTestOptions``, pass the option class explicitly::

  lbexec --override-option-class=Allen.config:AllenTestOptions <options_module.py>:main <options.yaml>

Examples of Moore HLT1 options and their YAML files live under
``Moore/Hlt/Hlt1Conf/options/`` and the corresponding pytest tests under
``Moore/Hlt/Hlt1Conf/tests/pytest/``.  For physics studies within Moore, see
:ref:`moore_performance_scripts`.

.. _run_allen_online:

Running Allen online (data-taking)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Online (and throughput) running is done through the MooreOnline testbench.  The
same ``MultiEventScheduler`` is configured by
``MooreOnline/AllenOnline/options/AllenConfig.py``, which:

* builds an ``AllenOptions`` object with ``input_type="MEP"``,
* injects an ``MEPProvider`` reading from MBM (or files/MPI for tests),
* selects the sequence from ``--hlt-type`` or from a TCK,
* configures the scheduler with ``create_appMgr.global_bind(make_scheduler=make_MultiEventScheduler)``,
* sets up the online monitoring and output services.

A non-interactive testbench run looks like::

  MooreOnline/run MooreOnline/MooreScripts/scripts/testbench.py \
    --working-dir=hlt1slim \
    MooreOnline/MooreScripts/tests/options/HLT1Slim/Arch.xml \
    --test-file-db-key=2024_mep_292860_run_change_test \
    --hlt-type=hlt1_pp_no_ut \
    --tfdb-nfiles 2 \
    --measure-throughput=0

Throughput measurements must be performed exclusively through the MooreOnline
testbench; see :ref:`measuring_throughput`.

.. _find_a_used_sequence:

Finding a sequence used on data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finding the name of a sequence that has been used on a specific LHCb dataset
may be done using the `runDB <https://lbrundb.cern.ch/rundb/export>`_.  From
here the option ``Trigger Conf`` may be selected and the Allen sequence used
when running HLT1 for any Run 3 dataset may be found.

Searching may also be done over a specified time period rather than for
specific runs.

.. _run_allen_standalone:

Standalone Allen (deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   Standalone Allen mode is **deprecated** and will be removed in a subsequent
   set of merge requests.  Use :ref:`run_allen_in_stack` or
   :ref:`run_allen_online` instead.  The information below is kept for
   reference only.

Some input files are included with the project for testing:

* ``input/minbias/mdf/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf``: Minbias sample produced from MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster TestFile DB entry. Includes raw banks with MC information.
* The directory ``input/detector_configuration`` contains the dumped geometry files for MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster
* Other dumped Allen geometries are located in ``/scratch/allen_geometries`` in the LHCb Online domain, and are used for other data sets in the CI tests
* Dumped Allen geometries can also be found in eos under ``/eos/lhcb/wg/rta/WP6/Allen/geometries``

A run of the Allen program with the help option ``-h`` will let you know the basic options::

    Usage: ./Allen
     -g {folder containing detector configuration}=../input/detector_configuration/
     --mdf {comma-separated list of MDF files to use as input OR single text file containing one MDF file per line}
     --mep {comma-separated list of MEP files to use as input}
     --transpose-mep {Transpose MEPs instead of decoding from MEP layout directly}=0 (don't transpose)
     --print-status {show status of buffer and socket}=0
     --print-config {show current algorithm configuration}=0
     --write-configuration {write current algorithm configuration to file}=0
     -n, --number-of-events {number of events to process}=0 (all)
     -s, --number-of-slices {number of input slices to allocate}=0 (one more than the number of threads)
     --events-per-slice {number of events per slice}=1000
     -t, --threads {number of threads / streams}=1
     -r, --repetitions {number of repetitions per thread / stream}=1
     -m, --memory {memory to reserve on the device per thread / stream (megabytes)}=1000
     --host-memory {memory to reserve on the host per thread / stream (megabytes)}=200
     -v, --verbosity {verbosity [0-5]}=3 (info)
     -p, --print-memory {print memory usage}=0
     --sequence {sequence to run}
     --output-file {Write selected event to output file}
     --device {select device to use}=0
     --non-stop {Runs the program indefinitely}=0
     --with-mpi {Read events with MPI}
     --mpi-window-size {Size of MPI sliding window}=4
     --mpi-number-of-slices {Number of MPI network slices}=6
     --inject-mem-fail {Whether to insert random memory failures (0: off 1-15: rate of 1 in 2^N)}=0
     --monitoring-filename {ROOT file to write monitoring histograms to}=monitoring.root
     --monitoring-save-period {Number of seconds between writes of the monitoring histograms (0: off)}=0
     --disable-run-changes {Ignore signals to update non-event data with each run change}=1
     -h {show this help}

Here are some examples for run options.  Note that if Allen was
:ref:`built with cvmfs<build with cvmfs>`, one can prepend
``./toolchain/wrapper`` to all the following commands to execute in the
correct environment.  ::

    # Run on an MDF input file shipped with Allen once
    ./Allen --sequence hlt1_pp_default --mdf ../input/minbias/mdf/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf

    # Run a total of 1000 events once with validation
    ./Allen --sequence hlt1_pp_validation -n 1000 --mdf /path/to/mdf/input/file

    # Run four streams, each with 4000 events and 20 repetitions
    ./Allen --sequence hlt1_pp_default -t 4 -n 4000 -r 20 --mdf /path/to/mdf/input/file

    # Run one stream with 5000 events and print all memory allocations
    ./Allen --sequence hlt1_pp_default -n 5000 -p 1 --mdf /path/to/mdf/input/file

    # Run on all events in all files listed in file.lst; four streams
    # with batches of 1000 events
    find /some/directory/with/files -type f | sort > files.lst
    ./Allen --sequence hlt1_pp_default -t 4 --events-per-slice 1000 --mdf /path/to/files.lst
