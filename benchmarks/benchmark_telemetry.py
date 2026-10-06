#!/usr/bin/env python3
"""Sample GPU and host state during a benchmark; never change device settings."""
import argparse
import csv
import datetime as dt
import io
import json
from pathlib import Path
import re
import signal
import subprocess
import time

GPU_FIELDS = (
    "uuid", "pstate", "clocks.current.sm", "clocks.current.memory",
    "utilization.gpu", "utilization.memory", "temperature.gpu", "power.draw",
    "power.limit", "memory.used", "clocks_event_reasons.active",
    "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def rows(text):
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(text), skipinitialspace=True) if row]


def cpu_counters():
    counters = {}
    for line in read("/proc/stat").splitlines():
        if re.match(r"cpu\d+ ", line):
            key, *fields = line.split()
            values = [int(v) for v in fields[:8]]  # guest time is already in user/nice
            counters[int(key[3:])] = (sum(values), values[3] + values[4])
    return counters


def cpu_busy(previous, current):
    return {str(cpu): 100 * (1 - (idle - previous[cpu][1]) / (total - previous[cpu][0]))
            for cpu, (total, idle) in current.items()
            if cpu in previous and total > previous[cpu][0]}


def process_info(pid):
    base = Path("/proc") / str(pid)
    status = read(base / "status")
    affinity = re.search(r"^Cpus_allowed_list:\s*(.*)$", status, re.M)
    runtime = wait = switches = 0
    cpus = []
    try:
        tasks = list((base / "task").iterdir())
    except OSError:
        tasks = []
    for task in tasks:
        stat = read(task / "stat").rpartition(")")[2].split()
        sched = read(task / "schedstat").split()
        if len(stat) > 36:
            cpus.append(int(stat[36]))  # processor, field 39
        if len(sched) == 3:
            a, b, c = map(int, sched)
            runtime += a; wait += b; switches += c
    pages = {}
    for node, count in re.findall(r"\bN(\d+)=(\d+)", read(base / "numa_maps")):
        pages[node] = pages.get(node, 0) + int(count)
    return {"pid": pid, "comm": read(base / "comm").strip(),
            "affinity": affinity[1] if affinity else None, "threads": len(tasks),
            "last_cpu_per_thread": cpus, "runtime_ns": runtime, "runqueue_wait_ns": wait,
            "timeslices": switches, "numa_pages": pages}


def sample(device, previous):
    gpu_rows = rows(command(["nvidia-smi", "-i", str(device),
                             "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"]))
    gpu = dict(zip(GPU_FIELDS, gpu_rows[0])) if gpu_rows else {}
    apps = rows(command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                         "--format=csv,noheader,nounits"]))
    apps = [dict(zip(("uuid", "pid", "name", "memory_mib"), row)) for row in apps
            if len(row) == 4 and row[0] == gpu.get("uuid")]
    current = cpu_counters()
    processes = [process_info(int(app["pid"])) for app in apps if app["pid"].isdigit()]
    return {"time_utc": now(), "monotonic_s": time.monotonic(), "gpu": gpu,
            "gpu_processes": apps, "processes": processes,
            "loadavg": read("/proc/loadavg").strip(), "cpu_busy_pct": cpu_busy(previous, current)}, current


def mark(directory, phase, repeat, role, log):
    event = {"time_utc": now(), "monotonic_s": time.monotonic(), "phase": phase,
             "repeat": repeat, "role": role, "log": str(Path(log).resolve()) if log else None}
    temporary = directory / "telemetry_state.tmp"
    temporary.write_text(json.dumps(event) + "\n")
    temporary.replace(directory / "telemetry_state.json")
    with (directory / "telemetry_events.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    mark_parser = sub.add_parser("mark")
    mark_parser.add_argument("directory", type=Path)
    mark_parser.add_argument("phase")
    mark_parser.add_argument("repeat", type=int)
    mark_parser.add_argument("role")
    mark_parser.add_argument("log", nargs="?", default="")
    monitor = sub.add_parser("monitor")
    monitor.add_argument("directory", type=Path)
    monitor.add_argument("--device", type=int, required=True)
    monitor.add_argument("--parent", type=int, required=True)
    monitor.add_argument("--interval", type=float, default=2)
    monitor.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.action == "mark":
        mark(args.directory, args.phase, args.repeat, args.role, args.log)
        return
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous = cpu_counters()
    with (args.directory / "telemetry.jsonl").open("a", buffering=1) as stream:
        while not stopping and Path(f"/proc/{args.parent}").exists():
            tick = time.monotonic()
            record, previous = sample(args.device, previous)
            try:
                state = json.loads(read(args.directory / "telemetry_state.json"))
            except ValueError:
                state = {}
            record["phase"] = state
            # Line-buffered Allen output lets analysis isolate its timer window.
            log = read(state.get("log", "")) if state.get("log") else ""
            record["timer_started"] = "Starting timer for throughput measurement" in log
            record["processing_complete"] = "Processing complete" in log
            stream.write(json.dumps(record) + "\n")
            if args.once:
                break
            while not stopping and time.monotonic() - tick < args.interval:
                time.sleep(0.1)


if __name__ == "__main__":
    main()
