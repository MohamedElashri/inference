# Allen — LHCb GPU HLT1 Trigger

**Depends on:** Rec (→ Lbcom → LHCb → Gaudi, Detector)

[![pipeline status](https://gitlab.cern.ch/lhcb/Allen/badges/master/pipeline.svg)](https://gitlab.cern.ch/lhcb/Allen/-/commits/master)

LHCb's GPU-accelerated first-stage trigger (HLT1). Allen processes ~30 MHz of proton-proton collisions and filters to ~1 MHz for HLT2 (Moore).

Full documentation: https://allen-doc.docs.cern.ch/index.html

---

## Resources

- **Documentation**: https://allen-doc.docs.cern.ch/index.html
- **Throughput evolution**: https://lbgrafana.cern.ch/d/Qvm54N3Mz/allen-performance?orgId=1
- **Physics performance dashboard**: https://lblhcbpr.cern.ch/dashboards/allen
- **Allen developers** (Mattermost): https://mattermost.web.cern.ch/lhcb/channels/allen-developers
- **Allen core** (Mattermost): https://mattermost.web.cern.ch/lhcb/channels/allen-core
- **AllenPR throughput** (Mattermost): https://mattermost.web.cern.ch/lhcb/channels/allenpr-throughput

---

## Contributing

**Active branches:**

| Branch          | Purpose                                              |
|-----------------|------------------------------------------------------|
| `master`        | Long-term development                                |
| `202X-patches`  | Fixes and additions for 202X data-taking             |

All protected branches require a merge request.

---

## Dependencies

```text
LCG (external: ROOT, Boost, …)  [via CVMFS]
└── Gaudi          ← application framework
    ├── Detector   ← DD4hep geometry and conditions
    └── LHCb       ← core event model, detector interfaces
        └── Lbcom  ← detector-specific algorithms and MC linkers
            └── Rec      ← CPU reconstruction algorithms
                └── Allen  ← this package
```
