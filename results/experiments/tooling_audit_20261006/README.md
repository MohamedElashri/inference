# Tooling and documentation audit, 6 October 2026

Audited active tracked project documentation, scripts and PVFinder integration
references against the current workflow and Allen APIs.

Fixed:

- Converter/verification output and model-shape diagnostics advertised removed
  `ballen` flags. They now identify the shared build script and its environment
  settings; the existing diagnostic test checks the new message.
- FC/UNet standalone validators silently selected the former default checkpoint;
  UNet also selected an obsolete dump directory. Dump and checkpoint arguments
  are now explicit, as in the full-model validator. The weights pipeline already
  supplies them.
- The reduced-work sequence generator used the old dictionary return API and
  masked VELO module 21, which the current baseline does not mask. It now uses
  the HLT1 user hook, reuses VELO instances and wraps PVFinder in the same physics
  prefilters. All 26 generated files were refreshed and configured successfully.
- The MC comparison description still claimed to run a removed standalone
  checker. It now explains direct MC-PV bank input and the retained matching rules.
- Run-record documentation described two weight binaries and misclassified the
  update-verification record. It now describes the tensor JSON and the separate
  verification schema. Record display only prints the historical cuBLAS option
  when that option was recorded.
- Integration comments now distinguish the FC/UNet benchmark from the full PV
  shadow chain and explain that BF16 uses fused CUDA kernels. Script help reflects
  the default build path and the FP32 override.

Checks: incremental build, 14 GPU test cases/314 assertions, all 26 generated
sequence configurations, repeated-update integration test, Python and shell
syntax, active Markdown links, checkpoint/model verification, and both changed
numerical-validator commands passed. The validators used an existing BF16 dump;
no fresh throughput or full MC run was needed for these documentation, diagnostic
and auxiliary-generator changes. `verification.json` records the checks and
source hashes, with compact supporting reports alongside it.

The active tracked-file scan found no remaining removed build flags, obsolete
build directories or version-specific workflow scripts. The FP32 model catalog
entries and the algorithms' ParamFiles fallback are supported paths. Historical
release notes, experiment records and the legacy run importer retain their
original references. Ignored local campaign archives and old build/data caches
remain on disk; the active documented workflow does not use their scripts.
