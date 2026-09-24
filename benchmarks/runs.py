#!/usr/bin/env python3
"""Tracked run records for PVFinder benchmarks, profiles and validations.

Every Allen run driven by benchmarks/benchmark_pvfinder_batch.sh or
`make -C weights validate` leaves one JSON file under results/runs/, tracked
in git. The raw logs, configs and nsys reports stay in the (ignored)
benchmark_results/ batch directory the record points to.

  runs.py snapshot BATCH_DIR --device N --build-dir DIR --model NAME
      Capture the environment at the start of a batch (host, GPU, git, build,
      model) into BATCH_DIR/snapshot.json.
  runs.py record BATCH_DIR [--status ok|failed] [--kind benchmark|profile]
      Combine the snapshot with the batch's results into results/runs/<id>.json.
  runs.py import BATCH_DIR...
      Same as record for batches that predate snapshots (best effort,
      marked "imported").
  runs.py validation --model NAME --build-dir DIR --dump-dir DIR --device N
                     --fc-report F --unet-report U [--label L]
      Record a validation (dump + validate_fc/validate_unet) run.
  runs.py list [--kind K] [--model M] [--gpu G] [--label S] [--all]
  runs.py show RUN
  runs.py compare RUN_A RUN_B

RUN is a record id, a unique id prefix/substring, or a path to a record.
Standard library only, so it runs under any python3.
"""

import argparse
import csv
import datetime as dt
import glob
import hashlib
import io
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys

SCHEMA = "pvfinder-run/1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, "results", "runs")
CATALOG = os.path.join(REPO_ROOT, "weights", "models.tsv")

# CMake cache entries worth keeping: they decide what the binary computes.
CMAKE_KEYS = (
    "CMAKE_BUILD_TYPE", "CUDA_ARCH", "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_COMPILER",
    "CMAKE_TOOLCHAIN_FILE", "WITH_CUDNN", "CUDNN_VERSION", "WITH_CUBLAS",
    "PVFINDER_UNET_N_FEAT", "PVFINDER_UNET_N_BATCH_CHANNELS", "BUILD_TESTING",
)
# Kernels always kept in a profile summary, whatever their rank.
PVFINDER_KERNEL_RE = re.compile(r"pvfinder|unet|cudnn|cublas|gemm|conv|softplus|maxpool|bf16|f16|sm80|sm86|sm75",
                                re.IGNORECASE)
TOP_KERNELS = 25
SEQUENCE_KEYS = ("baseline", "fc", "unet")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def run(cmd, cwd=None):
    """stdout of cmd, or None if it cannot run."""
    try:
        return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            return fp.read()
    except OSError:
        return None


def read_json(path):
    text = read_text(path)
    return json.loads(text) if text else None


def read_env(path):
    """key=value lines (metadata.env); comments and blanks skipped."""
    out = {}
    for line in (read_text(path) or "").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def typed(v):
    """metadata.env values back to bool/int/float where they clearly are one."""
    if v in ("true", "false"):
        return v == "true"
    if v == "" or v.startswith("<"):
        return None
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


def rel(path):
    if not path:
        return path
    ap = os.path.abspath(path)
    return os.path.relpath(ap, REPO_ROOT) if ap.startswith(REPO_ROOT + os.sep) else ap


def slug(text):
    return re.sub(r"[^A-Za-z0-9.=-]+", "-", text).strip("-").lower()


def gpu_slug(name):
    """'NVIDIA GeForce RTX 3090' -> 'rtx3090'"""
    if not name:
        return "unknown-gpu"
    s = re.sub(r"^(nvidia\s+)?(geforce\s+)?", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", s.lower()) or "unknown-gpu"


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


# ---------------------------------------------------------------------------
# environment capture
# ---------------------------------------------------------------------------
def host_info():
    cpu = None
    for line in (read_text("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    return {"hostname": socket.gethostname(), "os": platform.platform(), "cpu": cpu,
            "cpu_count": os.cpu_count(), "user": os.environ.get("USER")}


def gpu_info(device):
    fields = ["index", "name", "uuid", "driver_version", "compute_cap", "memory.total",
              "pci.bus_id", "clocks.max.sm", "power.limit"]
    out = run(["nvidia-smi", "-i", str(device), "--query-gpu=" + ",".join(fields),
               "--format=csv,noheader,nounits"])
    if not out:
        return {"index": device}
    vals = [v.strip() for v in next(csv.reader(io.StringIO(out.strip())))]
    info = dict(zip(fields, vals))
    gpu = {
        "index": int(info["index"]), "name": info["name"], "uuid": info["uuid"],
        "driver": info["driver_version"], "compute_capability": info["compute_cap"],
        "memory_total_mib": typed(info["memory.total"]), "pci_bus_id": info["pci.bus_id"],
        "max_sm_clock_mhz": typed(info["clocks.max.sm"]), "power_limit_w": typed(info["power.limit"]),
    }
    # Other compute processes on this GPU when the batch starts: contention.
    apps = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits"]) or ""
    gpu["processes_at_start"] = [
        {"pid": typed(r[1].strip()), "name": r[2].strip(), "used_memory_mib": typed(r[3].strip())}
        for r in csv.reader(io.StringIO(apps)) if len(r) == 4 and r[0].strip() == gpu["uuid"]
    ]
    return gpu


def git_info():
    head = (run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT) or "").strip() or None
    status = run(["git", "status", "--porcelain"], cwd=REPO_ROOT) or ""
    # Run records themselves are not code: a new record does not make the tree dirty.
    dirty = [line[3:] for line in status.splitlines() if line.strip() and not line[3:].startswith("results/")]
    diff = subprocess.run(["git", "diff", "HEAD", "--binary", "--", ".", ":(exclude)results"], cwd=REPO_ROOT,
                          capture_output=True).stdout if head else b""
    return {
        "head": head,
        "branch": (run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT) or "").strip() or None,
        "subject": (run(["git", "log", "-1", "--format=%s"], cwd=REPO_ROOT) or "").strip() or None,
        "dirty": bool(dirty),
        "dirty_files": dirty[:200],
        # Hash of the uncommitted tracked changes: two runs with the same head
        # and diff hash ran the same code.
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def cmake_cache(build_dir):
    out = {}
    for line in (read_text(os.path.join(build_dir, "CMakeCache.txt")) or "").splitlines():
        m = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", line)
        if m and m.group(1) in CMAKE_KEYS:
            out[m.group(1)] = m.group(2)
    for lang in ("CUDA", "CXX"):
        for f in glob.glob(os.path.join(build_dir, "CMakeFiles", "*", f"CMake{lang}Compiler.cmake")):
            m = re.search(rf'set\(CMAKE_{lang}_COMPILER_VERSION "([^"]+)"\)', read_text(f) or "")
            if m:
                out[f"CMAKE_{lang}_COMPILER_VERSION"] = m.group(1)
                break
    return out


def build_info(build_dir):
    lib = os.path.join(build_dir, "libAllenLib.so")
    mtime = os.path.getmtime(lib) if os.path.isfile(lib) else None
    info = {
        "name": os.path.basename(build_dir.rstrip("/")),
        "dir": rel(build_dir),
        "cmake": cmake_cache(build_dir),
        "lib_mtime": dt.datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds") if mtime else None,
    }
    # Tracked Allen sources edited after the library was linked. mtime-based,
    # so a checkout can raise false alarms, but an empty list with a clean tree
    # means the binary was built from what is checked in.
    if mtime:
        files = (run(["git", "ls-files", "--", "Allen/device", "Allen/host", "Allen/backend",
                      "Allen/stream", "Allen/main", "Allen/integration"], cwd=REPO_ROOT) or "").splitlines()
        newer = [f for f in files if os.path.getmtime(os.path.join(REPO_ROOT, f)) > mtime]
        info["sources_newer_than_build"] = newer[:50]
    return info


def catalog_row(model):
    text = read_text(CATALOG) or ""
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    if not rows:
        return None
    header = rows[0]
    for r in rows[1:]:
        if r and r[0] == model:
            return dict(zip(header, r))
    return None


def model_info(model, cnn_weights=None, fc_weights=None):
    out_dir = os.path.join(REPO_ROOT, "weights", "out", model) if model else None
    cnn = cnn_weights or (os.path.join(out_dir, "cnn_weights.bin") if out_dir else None)
    fc = fc_weights or (os.path.join(out_dir, "fc_weights.bin") if out_dir else None)
    ckpt = os.path.join(REPO_ROOT, "weights", "checkpoints", f"{model}.pyt") if model else None
    verify = read_text(os.path.join(out_dir, "verify.txt")) if out_dir else None
    return {
        "name": model,
        "catalog": catalog_row(model) if model else None,
        "checkpoint": {"path": rel(ckpt), "sha256": sha256_file(ckpt)},
        "weights": {
            "cnn": {"path": rel(cnn), "sha256": sha256_file(cnn)},
            "fc": {"path": rel(fc), "sha256": sha256_file(fc)},
        },
        "verified": bool(verify) and "FAIL" not in verify,
    }


def cmd_snapshot(args):
    snap = {
        "started_at": now_iso(),
        "host": host_info(),
        "gpu": gpu_info(args.device),
        "git": git_info(),
        "build": build_info(args.build_dir),
        "model": model_info(args.model, args.cnn_weights, args.fc_weights),
    }
    with open(os.path.join(args.batch_dir, "snapshot.json"), "w") as fp:
        json.dump(snap, fp, indent=2)
        fp.write("\n")


# ---------------------------------------------------------------------------
# batch results
# ---------------------------------------------------------------------------
def read_rates(run_dir):
    rates = {}
    for line in (read_text(os.path.join(run_dir, "rates.tsv")) or "").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            rates[parts[0]] = float(parts[1])
    return rates


def read_splits(run_dir):
    splits = {}
    for line in (read_text(os.path.join(run_dir, "slice_splits.tsv")) or "").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            splits[parts[0]] = int(parts[1])
    return splits


def derived(rates):
    b, f, u = (rates.get(k) for k in SEQUENCE_KEYS)
    out = {}
    if b:
        if f is not None:
            out["fc_overhead_pct"] = (b - f) / b * 100.0
        if u is not None:
            out["unet_overhead_pct"] = (b - u) / b * 100.0
            out["unet_retention_pct"] = u / b * 100.0
    return out


def pvfinder_config(run_dir):
    """The pvfinder_* algorithm blocks of each sequence's effective config."""
    out = {}
    for seq in SEQUENCE_KEYS:
        path = os.path.join(run_dir, f"{seq}_effective_config.json")
        if not os.path.isfile(path):
            continue
        cfg = read_json(path) or {}
        out[seq] = {
            "sha256": sha256_file(path),
            "algorithms": relativize({k: v for k, v in sorted(cfg.items()) if k.startswith("pvfinder")}),
        }
    return out


def relativize(obj):
    """Repository paths inside config values, made relative to the repository."""
    if isinstance(obj, dict):
        return {k: relativize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativize(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(REPO_ROOT + os.sep):
        return rel(obj)
    return obj


def parse_kernel_csv(path):
    """nsys stats cuda_gpu_kern_sum CSV -> list of kernel dicts."""
    text = read_text(path)
    if not text:
        return None
    # nsys may print notices before the header line.
    start = text.find('"Time (%)"')
    if start < 0:
        start = text.find("Time (%)")
    if start < 0:
        return None
    rows = []
    for r in csv.DictReader(io.StringIO(text[start:])):
        try:
            rows.append({
                "name": (r.get("Name") or "")[:240],
                "time_pct": float(r["Time (%)"]),
                "total_ns": int(float(r["Total Time (ns)"])),
                "instances": int(float(r["Instances"])),
                "avg_ns": float(r["Avg (ns)"]),
                "med_ns": float(r["Med (ns)"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return rows


def profile_summary(batch_dir, run_dirs):
    """Per sequence: kernel summary of each profiled repeat, merged by median."""
    per_seq = {}
    for seq in SEQUENCE_KEYS:
        tables = []
        for rd in run_dirs:
            for path in sorted(glob.glob(os.path.join(rd, f"pvfinder_profile_{seq}*cuda_gpu_kern_sum*.csv"))):
                t = parse_kernel_csv(path)
                if t:
                    tables.append(t)
        if not tables:
            continue
        merged = {}
        for t in tables:
            for k in t:
                merged.setdefault(k["name"], []).append(k)
        kernels = []
        for name, ks in merged.items():
            kernels.append({
                "name": name,
                "total_ns": median([k["total_ns"] for k in ks]),
                "instances": median([k["instances"] for k in ks]),
                "avg_ns": median([k["avg_ns"] for k in ks]),
                "time_pct": median([k["time_pct"] for k in ks]),
                "repeats_seen": len(ks),
            })
        kernels.sort(key=lambda k: k["total_ns"] or 0, reverse=True)
        kept = [k for i, k in enumerate(kernels) if i < TOP_KERNELS or PVFINDER_KERNEL_RE.search(k["name"])]
        per_seq[seq] = {
            "profiled_repeats": len(tables),
            "gpu_kernel_time_ns": median([sum(k["total_ns"] for k in t) for t in tables]),
            "n_kernel_names": len(kernels),
            "kernels": kept,
        }
    return {"tool": "nsys", "report": "cuda_gpu_kern_sum", "sequences": per_seq} if per_seq else None


def results_block(batch_dir):
    run_dirs = sorted(d for d in glob.glob(os.path.join(batch_dir, "run_*")) if os.path.isdir(d))
    repeats = []
    for rd in run_dirs:
        rates = read_rates(rd)
        if not rates:
            continue
        entry = {"run": int(os.path.basename(rd).split("_")[1]), "events_per_s": rates}
        entry.update(derived(rates))
        splits = read_splits(rd)
        if splits:
            entry["slice_splits"] = splits
        repeats.append(entry)
    summary = {}
    if repeats:
        med = {k: median([r["events_per_s"].get(k) for r in repeats]) for k in SEQUENCE_KEYS}
        summary["median_events_per_s"] = {k: v for k, v in med.items() if v is not None}
        for key in ("fc_overhead_pct", "unet_overhead_pct", "unet_retention_pct"):
            v = median([r.get(key) for r in repeats])
            if v is not None:
                summary[f"median_{key}"] = v
        base = [r["events_per_s"]["baseline"] for r in repeats if "baseline" in r["events_per_s"]]
        if base and med.get("baseline"):
            spread = (max(base) - min(base)) / med["baseline"] * 100.0
            summary["baseline_spread_pct"] = spread
            summary["contention"] = "contended" if spread > 5.0 else "acceptable"
        summary["slice_splits"] = any("slice_splits" in r for r in repeats)
    return run_dirs, {"units": "events/s", "repeats": repeats, "summary": summary}


def workload_block(meta):
    return {
        "mdf": rel(meta.get("mdf")),
        "geometry": rel(meta.get("geometry")),
        "events": typed(meta.get("events", "")),
        "memory_mb": typed(meta.get("memory", "")),
        "repetitions": typed(meta.get("repetitions", "")),
        "threads": typed(meta.get("threads", "")),
        "repeats": typed(meta.get("repeats", "")),
        "device": typed(meta.get("device", "")),
        "sequences": meta.get("sequences", "").split() or None,
    }


WORKLOAD_KEYS = {"events", "memory", "repetitions", "threads", "repeats", "device", "mdf", "geometry",
                 "sequences", "label", "timestamp", "build_name", "build_dir", "profile", "model",
                 "cnn_weights", "fc_weights"}


def legacy_snapshot(batch_dir, meta):
    """Best-effort environment for batches recorded before snapshot.json existed."""
    smi = read_text(os.path.join(batch_dir, "nvidia_smi.txt")) or ""
    device = typed(meta.get("device", ""))
    gpu = {"index": device}
    # nvidia-smi table rows: "|   2  NVIDIA GeForce RTX 3090   Off | ..."
    for line in smi.splitlines():
        m = re.match(r"\|\s+(\d+)\s+(NVIDIA[^|]*?)\s+(On|Off)\s+\|", line)
        if m and int(m.group(1)) == device:
            gpu["name"] = m.group(2).strip()
    m = re.search(r"Driver Version:\s*([\d.]+)", smi)
    if m:
        gpu["driver"] = m.group(1)
    head = (read_text(os.path.join(batch_dir, "git_head.txt")) or "").strip() or None
    dirty = [l[3:] for l in (read_text(os.path.join(batch_dir, "git_status_short.txt")) or "").splitlines() if l.strip()]
    build_dir = meta.get("build_dir")
    build = build_info(build_dir) if build_dir and os.path.isdir(build_dir) else {"name": meta.get("build_name")}
    build.pop("sources_newer_than_build", None)  # about today's tree, not the batch's
    build.pop("lib_mtime", None)
    model = model_info(meta.get("model"), meta.get("cnn_weights"), meta.get("fc_weights")) if meta.get("model") else None
    # weights.sha256 was written at run time; prefer it over today's files.
    for line in (read_text(os.path.join(batch_dir, "weights.sha256")) or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and model:
            for key in ("cnn", "fc"):
                if os.path.basename(parts[1]) == f"{key}_weights.bin":
                    model["weights"][key] = {"path": rel(parts[1]), "sha256": parts[0]}
    started = None
    if meta.get("timestamp"):
        started = dt.datetime.strptime(meta["timestamp"], "%Y%m%d_%H%M%S").astimezone().isoformat(timespec="seconds")
    return {
        "started_at": started,
        "host": None,
        "gpu": gpu,
        "git": {"head": head, "dirty": bool(dirty), "dirty_files": dirty[:200], "diff_sha256": None},
        "build": build,
        "model": model,
    }


def record_path(record):
    parts = ["_".join(record["id"].split("_")[:2]),   # YYYYmmdd_HHMMSS
             gpu_slug((record.get("gpu") or {}).get("name")),
             slug((record.get("model") or {}).get("name") or "nomodel"),
             slug(record.get("label") or record["kind"])]
    return os.path.join(RUNS_DIR, "_".join(p for p in parts if p) + ".json")


def write_record(record):
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = record_path(record)
    with open(path, "w") as fp:
        json.dump(record, fp, indent=2)
        fp.write("\n")
    return path


def build_batch_record(batch_dir, status, kind=None, imported=False):
    batch_dir = os.path.abspath(batch_dir)
    meta = read_env(os.path.join(batch_dir, "metadata.env"))
    if not meta:
        raise SystemExit(f"{batch_dir}: no metadata.env, not a benchmark batch")
    snap = read_json(os.path.join(batch_dir, "snapshot.json"))
    if snap is None:
        if not imported:
            print(f"warning: {batch_dir} has no snapshot.json; using legacy files", file=sys.stderr)
        snap = legacy_snapshot(batch_dir, meta)
        imported = True
    run_dirs, results = results_block(batch_dir)
    profile = profile_summary(batch_dir, run_dirs) if meta.get("profile") == "1" else None
    if kind is None:
        kind = "profile" if meta.get("profile") == "1" else "benchmark"
    finished = None
    summary_md = os.path.join(batch_dir, "summary.md")
    if os.path.isfile(summary_md):
        finished = dt.datetime.fromtimestamp(os.path.getmtime(summary_md)).astimezone().isoformat(timespec="seconds")
    if not imported:
        finished = now_iso()
    record = {
        "schema": SCHEMA,
        "kind": kind,
        "id": os.path.basename(batch_dir),
        "label": meta.get("label"),
        "status": status,
        "imported": imported,
        "started_at": snap.get("started_at"),
        "finished_at": finished,
        "command": (read_text(os.path.join(batch_dir, "batch_command.cmd")) or "").strip() or None,
        "host": snap.get("host"),
        "gpu": snap.get("gpu"),
        "git": snap.get("git"),
        "build": snap.get("build"),
        "model": snap.get("model"),
        "workload": workload_block(meta),
        "options": {k: typed(v) for k, v in sorted(meta.items()) if k not in WORKLOAD_KEYS},
        "config": pvfinder_config(run_dirs[0]) if run_dirs else {},
        "results": results,
        "profile": profile,
        "artifacts": {"batch_dir": rel(batch_dir)},
    }
    if record["started_at"] and record["finished_at"]:
        t0 = dt.datetime.fromisoformat(record["started_at"])
        t1 = dt.datetime.fromisoformat(record["finished_at"])
        record["duration_s"] = round((t1 - t0).total_seconds())
    return record


def cmd_record(args):
    rec = build_batch_record(args.batch_dir, args.status, args.kind)
    path = write_record(rec)
    print(f"run record: {rel(path)}")


def cmd_import(args):
    for d in args.batch_dirs:
        try:
            rec = build_batch_record(d, "ok", imported=True)
        except SystemExit as e:
            print(e, file=sys.stderr)
            continue
        if not rec["results"]["repeats"]:
            print(f"{d}: no completed repeats, skipped", file=sys.stderr)
            continue
        print(f"run record: {rel(write_record(rec))}")


# ---------------------------------------------------------------------------
# validation runs
# ---------------------------------------------------------------------------
def cmd_validation(args):
    started = now_iso()
    fc = read_json(args.fc_report) if args.fc_report else None
    unet = read_json(args.unet_report) if args.unet_report else None
    if unet:
        unet.pop("per_event_max_abs_diff", None)   # 500 numbers; the batch dir keeps them
    cfg = read_json(os.path.join(args.dump_dir, "config.json")) or {}
    # The Allen side of a validation is the dump; `make dump` snapshots the
    # environment it ran in. Without a snapshot, describe the current one.
    snap = read_json(os.path.join(args.dump_dir, "snapshot.json")) or {
        "started_at": None, "host": host_info(), "gpu": gpu_info(args.device), "git": git_info(),
        "build": build_info(args.build_dir), "model": model_info(args.model)}
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or "validate"
    ok = all(r is None or r.get("status") == "PASS" for r in (fc, unet)) and (fc or unet)
    record = {
        "schema": SCHEMA,
        "kind": "validation",
        "id": f"{stamp}_{slug(label)}",
        "label": label,
        "status": "ok" if ok else "failed",
        "imported": False,
        "started_at": snap.get("started_at") or started,
        "finished_at": now_iso(),
        "host": snap.get("host"),
        "gpu": snap.get("gpu"),
        "git": snap.get("git"),
        "build": snap.get("build"),
        "model": snap.get("model"),
        "workload": {"events": args.events, "threads": 1, "device": args.device,
                     "sequence": args.sequence},
        "config": {"dump": {"algorithms": relativize({k: v for k, v in sorted(cfg.items()) if k.startswith("pvfinder")})}},
        "results": {"fc": fc, "unet": unet},
        "profile": None,
        "artifacts": {"dump_dir": rel(args.dump_dir)},
    }
    t0 = dt.datetime.fromisoformat(record["started_at"])
    record["duration_s"] = round((dt.datetime.fromisoformat(record["finished_at"]) - t0).total_seconds())
    print(f"run record: {rel(write_record(record))}")
    return 0


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------
def load_records():
    recs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        r = read_json(path)
        if r and r.get("schema", "").startswith("pvfinder-run/"):
            r["_path"] = path
            recs.append(r)
    return recs


def find_record(key):
    if os.path.isfile(key):
        r = read_json(key)
        r["_path"] = key
        return r
    recs = load_records()
    exact = [r for r in recs if r["id"] == key or os.path.basename(r["_path"]) in (key, key + ".json")]
    if exact:
        return exact[0]
    hits = [r for r in recs if key in r["id"] or key in os.path.basename(r["_path"])]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"'{key}' matches {len(hits)} records" + (": " + ", ".join(r["id"] for r in hits[:8]) if hits else ""))


def headline(r):
    s = (r.get("results") or {}).get("summary") or {}
    m = s.get("median_events_per_s") or {}
    if r["kind"] == "validation":
        u = (r.get("results") or {}).get("unet") or {}
        f = (r.get("results") or {}).get("fc") or {}
        return f"fc {f.get('status', '-')}, unet {u.get('status', '-')} max|d| {u.get('max_abs_diff', float('nan')):.2e}"
    fmt = lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else "-"
    return f"base {fmt(m.get('baseline'))}  fc {fmt(m.get('fc'))}  unet {fmt(m.get('unet'))}"


def cmd_list(args):
    rows = []
    for r in load_records():
        gpu = (r.get("gpu") or {}).get("name") or ""
        model = (r.get("model") or {}).get("name") or ""
        if args.kind and r["kind"] != args.kind:
            continue
        if args.model and args.model not in model:
            continue
        if args.gpu and args.gpu.lower() not in gpu.lower():
            continue
        if args.label and args.label not in (r.get("label") or ""):
            continue
        if not args.all and r.get("status") != "ok":
            continue
        w = r.get("workload") or {}
        point = f"n{w.get('events')} m{w.get('memory_mb')} r{w.get('repetitions')} t{w.get('threads')}" \
            if r["kind"] != "validation" else f"n{w.get('events')}"
        rows.append((r["id"], r["kind"], r.get("status"), gpu_slug(gpu), model, point, headline(r)))
    if not rows:
        print("no matching records")
        return
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


def cmd_show(args):
    r = find_record(args.run)
    path = r.pop("_path")
    if args.json:
        print(json.dumps(r, indent=2))
        return
    g, b, m, w = (r.get(k) or {} for k in ("gpu", "build", "model", "workload"))
    print(f"{r['id']}  [{r['kind']}, {r.get('status')}{', imported' if r.get('imported') else ''}]  {rel(path)}")
    print(f"  started   {r.get('started_at')}  duration {r.get('duration_s', '-')} s")
    print(f"  gpu       {g.get('name')} (cc {g.get('compute_capability')}, driver {g.get('driver')}, index {g.get('index')})"
          + (f"; {len(g['processes_at_start'])} other process(es) at start" if g.get("processes_at_start") else ""))
    git = r.get("git") or {}
    print(f"  git       {str(git.get('head'))[:12]} {git.get('branch') or ''}{' (dirty)' if git.get('dirty') else ''}  {git.get('subject') or ''}")
    cm = b.get("cmake") or {}
    print(f"  build     {b.get('name')}  N_FEAT={cm.get('PVFINDER_UNET_N_FEAT')} latent={cm.get('PVFINDER_UNET_N_BATCH_CHANNELS')} "
          f"cuDNN={cm.get('CUDNN_VERSION')} cuBLAS={cm.get('WITH_CUBLAS')} CUDA={cm.get('CMAKE_CUDA_COMPILER_VERSION')} arch={cm.get('CUDA_ARCH')}")
    if b.get("sources_newer_than_build"):
        print(f"  WARNING   {len(b['sources_newer_than_build'])} tracked Allen source(s) newer than the build")
    print(f"  model     {m.get('name')}  cnn {str(((m.get('weights') or {}).get('cnn') or {}).get('sha256'))[:12]}"
          f"  fc {str(((m.get('weights') or {}).get('fc') or {}).get('sha256'))[:12]}")
    print(f"  workload  " + ", ".join(f"{k}={v}" for k, v in w.items() if v is not None and k not in ("mdf", "geometry")))
    opts = r.get("options") or {}
    if opts:
        print("  options   " + ", ".join(f"{k}={v}" for k, v in opts.items() if v is not None))
    print(f"  result    {headline(r)}")
    s = (r.get("results") or {}).get("summary") or {}
    for k in ("median_fc_overhead_pct", "median_unet_overhead_pct", "median_unet_retention_pct", "baseline_spread_pct"):
        if k in s:
            print(f"            {k} = {s[k]:.2f}")
    prof = r.get("profile")
    if prof:
        for seq, p in prof["sequences"].items():
            print(f"  profile   {seq}: {p['n_kernel_names']} kernels, GPU kernel time {p['gpu_kernel_time_ns'] / 1e6:.1f} ms")
            for k in p["kernels"][:8]:
                print(f"              {k['total_ns'] / 1e6:9.2f} ms  {int(k['instances']):7d}x  {k['name'][:90]}")


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# Bookkeeping that differs between any two runs; compare leaves it out.
COMPARE_SKIP = ("id", "label", "status", "imported", "started_at", "finished_at", "duration_s", "results",
                "profile", "artifacts", "command", "_path", "host", "gpu.processes_at_start", "gpu.uuid",
                "gpu.pci_bus_id", "gpu.index", "git.dirty_files", "git.subject", "build.lib_mtime",
                "build.sources_newer_than_build", "workload.device", "options.timestamp")


def compare_skipped(key):
    if any(key == s or key.startswith(s + ".") for s in COMPARE_SKIP):
        return True
    # A config file hash changes whenever one of its values does; the values
    # themselves are listed individually.
    return key.startswith("config.") and key.endswith(".sha256")


def cmd_compare(args):
    a, b = find_record(args.run_a), find_record(args.run_b)
    print(f"A = {a['id']}\nB = {b['id']}\n")
    ma = ((a.get("results") or {}).get("summary") or {}).get("median_events_per_s") or {}
    mb = ((b.get("results") or {}).get("summary") or {}).get("median_events_per_s") or {}
    if ma or mb:
        print(f"  {'sequence':10s} {'A ev/s':>12s} {'B ev/s':>12s} {'B/A':>8s}")
        for k in SEQUENCE_KEYS:
            va, vb = ma.get(k), mb.get(k)
            ratio = f"{vb / va:8.4f}" if va and vb else "       -"
            print(f"  {k:10s} {va or float('nan'):12,.1f} {vb or float('nan'):12,.1f} {ratio}")
        print()
    # What differs between the two runs, apart from bookkeeping.
    fa, fb = flatten(a), flatten(b)
    diffs = [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb))
             if fa.get(k) != fb.get(k) and not compare_skipped(k)]
    if not diffs:
        print("  no configuration differences")
    for k, va, vb in diffs:
        print(f"  {k}\n      A: {va}\n      B: {vb}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot")
    s.add_argument("batch_dir")
    s.add_argument("--device", type=int, required=True)
    s.add_argument("--build-dir", required=True)
    s.add_argument("--model", default=None)
    s.add_argument("--cnn-weights", default=None)
    s.add_argument("--fc-weights", default=None)
    s.set_defaults(func=cmd_snapshot)

    s = sub.add_parser("record")
    s.add_argument("batch_dir")
    s.add_argument("--status", default="ok", choices=["ok", "failed", "interrupted"])
    s.add_argument("--kind", default=None, choices=["benchmark", "profile"])
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("import")
    s.add_argument("batch_dirs", nargs="+")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("validation")
    s.add_argument("--model", required=True)
    s.add_argument("--build-dir", required=True)
    s.add_argument("--dump-dir", required=True)
    s.add_argument("--device", type=int, required=True)
    s.add_argument("--events", type=int, default=None)
    s.add_argument("--sequence", default=None)
    s.add_argument("--fc-report", default=None)
    s.add_argument("--unet-report", default=None)
    s.add_argument("--label", default=None)
    s.set_defaults(func=cmd_validation)

    s = sub.add_parser("list")
    s.add_argument("--kind", choices=["benchmark", "profile", "validation"])
    s.add_argument("--model")
    s.add_argument("--gpu")
    s.add_argument("--label")
    s.add_argument("--all", action="store_true", help="include failed/interrupted runs")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show")
    s.add_argument("run")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("compare")
    s.add_argument("run_a")
    s.add_argument("run_b")
    s.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
