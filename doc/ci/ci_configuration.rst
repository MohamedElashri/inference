Allen CI configuration
======================

.. warning::

   This page is **deprecated**.  The Allen CI configuration described below is
   being phased out; use the ``/ci-test`` GitLab command described in the next
   section to launch cross-project tests on a merge request.

Triggering a full cross-project test with ``/ci-test``
------------------------------------------------------

To run the full Allen CI on a merge request together with the interdependent
changes in other projects, post a comment on the merge request with::

  /ci-test LHCb!5683 Allen!2414 DaVinci!1564 Moore!6224 Panoptes!652 LHCbIntegrationTests!137 Online!1248

The ``/ci-test`` command launches a test of all the listed merge requests
together (for example ``LHCb!5683`` is LHCb merge request 5683,
``Allen!2414`` is Allen merge request 2414, and so on).  Only list the merge
requests that are actually interdependent; the CI then builds and tests the
resulting stack combination, including Allen's throughput tests.  The HLT1
physics validation (efficiency) tests have been migrated to Moore.

Deprecated CI configuration
---------------------------

The scripts to configure Allen's CI pipeline are located in `scripts/ci/config <https://gitlab.cern.ch/lhcb/Allen/-/tree/master/scripts/ci/config>`_
Two pipelines are defined and used as follows: Every commit to a merge request triggers the "minimal" pipeline. Before merging a merge request, the "full pipeline" with a larger varietey of build options and data sets is triggered manually from the merge request page.

Adding new devices
^^^^^^^^^^^^^^^^^^^^^^^^
1. Add an entry for the device to `devices.yaml <https://gitlab.cern.ch/lhcb/Allen/-/blob/master/scripts/ci/config/devices.yaml>`_. Set `TARGET`, `DEVICE_ID`, and the `tag:` accordingly

2. Add a job entry to run in the minimal pipeline: e.g.

.. code-block:: yaml

  epyc7502:
    extends:
      - .epyc7502
      - .run_job

3. Add a job entry to run in the full pipeline, taking care to `extends:` from the right key based on the `TARGET` of the device:

.. code-block:: yaml

  epyc7502-full:
    extends:
      - .epyc7502
      - .[cuda/hip/cpu]_run_job
      - .run_jobs_full

4. Add the job to the dependencies of `.depend_full_jobs`:

.. code-block:: yaml

  .depend_full_jobs:
    dependencies:
      - ...

5. If you added a new CUDA device, check `OVERRIDE_CUDA_ARCH_FLAG` in `.gitlab-ci.yml` contains the right flags for this device

Adding new tests
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   The test matrix is now defined in `scripts/ci/test_config.yaml` and executed
   by `scripts/ci/run_tests.py`; the legacy `parallel:matrix` description below
   is kept for reference only.

See `Gitlab CI documentation <https://docs.gitlab.com/ee/ci/yaml>`_ for more information on how the `parallel:matrix` keyword works.

To the minimal pipeline
-------------------------
Add a key to `.run_matrix_jobs_minimal:parallel:matrix:` in `common-run.yaml` e.g.

.. code-block:: yaml

      # throughput test
      - TEST_NAME: "run_throughput"       # name of the test
        SEQUENCES: ["hlt1_pp_default"]    # sequence(s) to run the test on
        DATA_TAG: ["Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD"] # input dataset

Other variables can be set (but are optional - see below).

To the full pipeline
-----------------------
Add a key to `.run_matrix_jobs_full:parallel:matrix:` in `common-run.yaml` e.g.

.. code-block:: yaml

      - TEST_NAME: "run_throughput"     # name of the test - runs the bash script scripts/ci/jobs/$TEST_NAME.sh
        LCG_OPTIMIZATION: ["opt"]       # use opt build
        # OPTIONS: [""]                 # leave out for default build, with no additional build options
        SEQUENCES: ["hlt1_pp_default"]  # sequence
        DATA_TAG: ["SMOG2_pppHe_retinacluster_v1"]  # dataset name
        # GEOMETRY: [""]                # don't add this, to use the default geom

If your test needs a build of Allen that is not yet included in the `build` stage, you will need to create one.

In order to ensure the correct build from the `build` stage is used in your test, make sure that the following variables are set correctly and match.

* `${LCG_SYSTEM}` (e.g. `x86_64_v3-el9-gcc12`. default value is set by `.run_jobs` key)
* `${LCG_QUALIFIER}` (added directly after `LCG_SYSTEM` with `+` delimiter - default is `cpu`)
* `${LCG_OPTIMIZATION}` (e.g. `opt` or `dbg`. default value is set in `.gitlab-ci.yaml` to `opt`)
* `${SEQUENCES}` (must be set in `.run_matrix_jobs_full:parallel:matrix:`)
* `${OPTIONS}` (optional, can be set in `.run_matrix_jobs_full:parallel:matrix:`)
* `${GEOMETRY}` (optional, can be left undefined or set if a specific geometry is needed)

Physics validation / efficiency tests
-----------------------------------------
The standalone HLT1 physics validation (efficiency) tests and their reference
files (`test/reference`) have been migrated to Moore. There the Allen HLT1
sequences are run and the reconstruction is validated with the Rec/Moore
checkers, instead of the Allen validators. They are no longer part of the Allen
CI.

The stack-level reference tests (`Rec/Allen/tests`) are still in use and are
updated via the reference-update bot. When a MR changes the stack references,
the bot opens a MR with the updated reference files; the update can also be
triggered from the bot's manual job.

Adding new builds
---------------------
The `parallel:matrix:` keys will need to be modified in either `.build_job_minimal_matrix` or `.build_job_additional_matrix`.

N.B.

* `$LCG_QUALIFIER` does not need to be set in `parallel:matrix:` for the full builds, but it will need to be for the minimal builds.
* `$OPTIONS` can be left blank or undefined. If options need to be passed to CMake e.g. `-DBUILD_TESTING=ON -DENABLE_CONTRACTS=ON`, then `$OPTIONS` can be set to `BUILD_TESTING+ENABLE_CONTRACTS` which will set both CMake options to `ON` by default. If you need this to be something other than `ON`, then you can do `BUILD_TESTING=OFF+ENABLE_CONTRACTS=OFF`, for example.
* In downstream `run`-stage jobs, the `$OPTIONS` variable content *must* match for the build to be found properly.
