"""Integration checks for repeated upstream updates and input preservation."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "allen_upstream.py"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.PIPE).decode().strip()


def initialize(path):
    path.mkdir()
    git(path, "init", "-b", "master")
    git(path, "config", "user.name", "test")
    git(path, "config", "user.email", "test@example.invalid")


def commit(path, message):
    git(path, "add", "-A")
    git(path, "commit", "-m", message)
    return git(path, "rev-parse", "HEAD")


class UpstreamUpdateTest(unittest.TestCase):
    def test_two_updates_replay_local_edits_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory(prefix="allen-update-test-") as directory:
            base = Path(directory)
            upstream, fork, root = (base / name for name in ("upstream", "fork", "inference"))
            initialize(upstream)
            (upstream / "api.txt").write_text("original upstream\n")
            (upstream / "input").mkdir()
            (upstream / "input/sample.dat").write_text("upstream sample\n")
            upstream_base = commit(upstream, "initial upstream")
            subprocess.run(["git", "clone", str(upstream), str(fork)], check=True, capture_output=True)
            git(fork, "config", "user.name", "test")
            git(fork, "config", "user.email", "test@example.invalid")
            (fork / "pvfinder.txt").write_text("original PVFinder\n")
            fork_commit = commit(fork, "PVFinder integration")
            initialize(root)
            (root / "Allen").mkdir()
            for name in ("api.txt", "pvfinder.txt"):
                shutil.copy2(fork / name, root / "Allen" / name)
            (root / "Allen/input").mkdir()
            local_input = root / "Allen/input/sample.dat"
            local_input.write_text("local detector data must survive\n")
            (root / ".gitignore").write_text("/Allen/input/\n/Allen/new_source.txt\n/benchmark_results/\n")
            (root / "benchmarks").mkdir()
            tracking = root / "benchmarks/allen_upstream.json"
            tracking.write_text(json.dumps({"upstream_commit": upstream_base, "fork_commit": fork_commit}))
            commit(root, "vendored inference source")
            git(root, "fetch", str(fork), "HEAD:refs/heads/allen/pvfinder")
            spec = importlib.util.spec_from_file_location("allen_upstream_under_test", SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.ROOT, module.TRACKING, module.URL = root, tracking, str(upstream)
            for iteration in (1, 2):
                (root / "Allen/pvfinder.txt").write_text(f"local PVFinder edit {iteration}\n")
                commit(root, f"PVFinder edit {iteration}")
                (upstream / "api.txt").write_text(f"upstream API {iteration}\n")
                if iteration == 1:
                    (upstream / "new_source.txt").write_text("new upstream source\n")
                    (root / "Allen/new_source.txt").write_text("new upstream source\n")
                target = commit(upstream, f"upstream update {iteration}")
                workdir = base / f"prepared{iteration}"
                before = git(root, "rev-parse", "HEAD")
                with contextlib.redirect_stdout(io.StringIO()):
                    module.prepare(SimpleNamespace(workdir=workdir, target="master"))
                self.assertEqual(git(root, "rev-parse", "HEAD"), before)
                manifest_path = workdir.parent / (workdir.name + ".update.json")
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["upstream_commit"], target)
                self.assertEqual((workdir / "pvfinder.txt").read_text(), f"local PVFinder edit {iteration}\n")
                proof = root / "results/verification.json"
                proof.parent.mkdir(exist_ok=True)
                proof.write_text(json.dumps({"status": "PASS", "fork_commit": "wrong revision"}))
                with self.assertRaises(SystemExit):
                    module.import_tree(SimpleNamespace(manifest=manifest_path, validation=proof))
                self.assertEqual(git(root, "rev-parse", "HEAD"), before)
                proof.write_text(json.dumps({"status": "PASS", "fork_commit": manifest["fork_commit"]}))
                with contextlib.redirect_stdout(io.StringIO()):
                    module.import_tree(SimpleNamespace(manifest=manifest_path, validation=proof))
                git(root, "commit", "-m", f"import {iteration}")
                self.assertEqual((root / "Allen/api.txt").read_text(), f"upstream API {iteration}\n")
                self.assertEqual((root / "Allen/pvfinder.txt").read_text(), f"local PVFinder edit {iteration}\n")
                self.assertEqual(local_input.read_text(), "local detector data must survive\n")
                self.assertEqual((root / "Allen/new_source.txt").read_text(), "new upstream source\n")
                self.assertEqual(git(root, "merge-base", "HEAD", manifest["fork_commit"]), manifest["fork_commit"])
                self.assertEqual(git(root, "status", "--porcelain", "--untracked-files=no"), "")


if __name__ == "__main__":
    unittest.main()
