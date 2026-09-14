.. _conditions:

Non-event data conditions
=========================

Allen needs detector geometry and conditions (collectively *non-event data*)
on the GPU in formats that are cheap to copy and use inside kernels.  These
formats are defined as **conditions** in
``integration/non_event_data/include/``, under the ``Allen::Conditions``
namespace.

Each condition has two equivalent representations:

* **derived from the conditions database** when Allen runs in the LHCb stack
  (Gaudi / DD4HEP or DetDesc).  This is the production path and is driven by
  ``addConditionDerivation``.
* **a binary dump** (a ``.bin`` file) produced by a *binary dumper*.  This is
  used by the standalone path (deprecated, see :ref:`run_allen_standalone`)
  and for producing/validating geometry dumps.

The condition object itself is shared by both paths: it knows how to fill
itself from Gaudi detector elements / conditions, how to read itself back from
a binary buffer, and how to copy its data into the global ``Constants``
structure that Allen algorithms read on the device.

Common building blocks
----------------------

``HostDeviceCondition<T>``
^^^^^^^^^^^^^^^^^^^^^^^^^^

``integration/non_event_data/include/HostDeviceCondition.h`` defines a small
RAII holder for a trivially-copyable, trivially-destructible POD condition
``T``.  It allocates both a host and a device copy of ``T`` and exposes
``host()`` / ``device()`` accessors plus a ``data()`` span for dumping::

  template<typename T>
  class HostDeviceCondition {
    HostDeviceCondition(T&& data);                  // from a POD
    HostDeviceCondition(const std::vector<char>&);  // from a binary dump
    T* host() const;
    T* device() const;
    std::span<const char> data() const;             // for binary dumps
  };

Conditions whose GPU representation is a single POD struct (for example
``VPGeometry``, ``UTGeometry``, ``SciFiGeometry``, ``EcalGeometry``,
``MagneticField``, ``MagneticFieldPolarity``, …) derive from it.

``ConstantsCondition`` and ``AllenUpdater``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``integration/non_event_data/include/ConstantsCondition.h`` defines
``Allen::Conditions::ConstantsCondition``.  Its ``addConditionDerivation``
registers a Gaudi condition derivation for every condition listed in its
dependency pack, and then derives the global ``Constants`` object that is
published at ``ConstantsCondition::DefaultLocation``.

The derivation lambda:

* calls ``updater->getConstants()`` to obtain the ``Constants`` instance,
* reads auxiliary parameter files from ``updater->getParamDir()``,
* calls ``dependency.update_constants(constants)`` for each condition to copy
  the condition data into the global ``Constants``,
* calls ``updater->dump(dependency)`` for each condition when binary dumping
  is enabled.

``AllenUpdater`` (``Dumpers/BinaryDumpers/include/Dumpers/AllenUpdater.h``) is
the Gaudi service that implements ``Allen::NonEventData::IUpdater``.  It owns
the ``Constants`` instance, exposes ``getConstants()`` / ``getParamDir()`` /
``getBeamlineOffset()``, and writes binary dumps through ``dump()``::

  template<typename T>
  std::size_t dump(const T& cond) {
    if (!m_dumpToFile) return 0;
    auto data = cond.data();
    auto filename = m_outputDirectory.value() + "/" + T::filename;
    std::ofstream output {filename, std::ios::out | std::ios::binary};
    output.write(data.data(), data.size());
    return data.size();
  }

Its ``OutputDirectory``, ``DumpToFile``, ``BeamlineOffset``, ``ParamDir`` and
``TriggerEventLoop`` properties configure the dumping behaviour.

Standalone reader (deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``integration/non_event_data/src/RegisterConsumers.cpp`` provides
``load_geometry``, which reads the binary ``.bin`` files back and calls
``update_constants`` on each condition.  This is used by the deprecated
standalone Allen event loop and is kept only for that path.

Writing a condition
-------------------

A condition lives in ``integration/non_event_data/include/``.  The steps are:

1. Define the condition struct in the ``Allen::Conditions`` namespace.

2. Provide the static metadata:

   * ``id`` — a unique identifier (``"VeloGeometry"``, ``"UTBoards"``, …),
   * ``filename`` — the name of the binary dump file (``velo_geometry.bin``, …),
   * ``DefaultLocation`` — the Gaudi condition location, with separate
     ``USE_DD4HEP`` and DetDesc variants.

3. Implement the Gaudi side with a static ``addConditionDerivation(PARENT*)``
   and a constructor taking the required detector elements / conditions.
   Inside the constructor, serialise into a ``std::vector<char>`` (directly or
   with ``DumpUtils::Writer``).

4. Implement the binary side with a ``std::vector<char>`` constructor that
   reconstructs the condition from a dump.

5. Implement ``update_constants(Constants&)`` (and, for larger
   variable-size data, ``initialize(Constants&)``) to copy the condition into
   the global ``Constants``, allocating device memory where needed.

6. Add the condition to the dependency pack in
   ``ConstantsCondition::addConditionDerivation`` (and to ``load_geometry``
   for the standalone path).

A simple POD condition derived from ``HostDeviceCondition`` looks like this
(abridged from ``integration/non_event_data/include/VPGeometry.h``)::

  namespace Allen::Conditions {
    struct VPGeometry : HostDeviceCondition<VeloGeometry> {
      inline static std::string const id = "VeloGeometry";
      inline static std::string const filename = "velo_geometry.bin";
  #ifdef USE_DD4HEP
      inline static std::string const DefaultLocation = "/world:AllenConditions-vp-geometry";
  #else
      inline static std::string const DefaultLocation = "AllenConditions-vp-geometry";
  #endif

      template<typename PARENT>
      static auto addConditionDerivation(PARENT* parent) {
        return parent->addConditionDerivation(
          {DeVPLocation::Default}, VPGeometry::DefaultLocation,
          [](DeVP const& det) { return Allen::Conditions::VPGeometry {det}; });
      }

      VPGeometry(const DeVP& det) : HostDeviceCondition<VeloGeometry>(/* fill VeloGeometry from det */) {}
      VPGeometry(const std::vector<char>& data) : HostDeviceCondition<VeloGeometry>(/* reconstruct from data */) {}

      void update_constants(Constants& constants) const {
        constants.dev_velo_geometry = device();
        constants.host_velo_geometry = host();
      }
    };
  }

Conditions that need more than a single POD (for example ``UTBoards``, which
builds several host/device maps) do not derive from ``HostDeviceCondition``.
Instead they keep a ``std::vector<char> m_data`` member, fill it in the Gaudi
constructor, reconstruct it in the ``std::vector<char>`` constructor, and
implement ``initialize()`` / ``update_constants()`` to populate
``Constants``.

Writing a binary dumper
-----------------------

Binary dumpers serialise the conditions into ``.bin`` files.  The serialisation
helper is ``DumpUtils::Writer``
(``Dumpers/BinaryDumpers/include/Dumpers/Utils.h``)::

  DumpUtils::Writer output {};
  output.write(version, someVector, somePOD, ...);
  m_data = output.buffer();

``write`` accepts any trivially-copyable value, span, or container that has
``LHCb::make_span``, and appends them in order to an in-memory buffer.

When ``AllenUpdater`` has ``DumpToFile`` set, the
``ConstantsCondition`` derivation lambda calls ``updater->dump(condition)``
for every condition, writing each ``T::filename`` into ``OutputDirectory``.
The dumped files are then available for the standalone reader or for
inspection/validation.

The condition itself decides the binary layout: it writes the same fields in
the Gaudi constructor (via ``DumpUtils::Writer``) and reads them back in the
``std::vector<char>`` constructor.  Version numbers are recommended so the
reader can stay backwards compatible (see the version handling in
``Beamline``).
