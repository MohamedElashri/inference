#!/usr/bin/env python3
"""Prepare an upstream rebase, then import a validated Allen tree into this repo."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
TRACKING = ROOT / "benchmarks/allen_upstream.json"
URL = "https://gitlab.cern.ch/lhcb/Allen.git"


def git(repo, *args, data=None, env=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], input=data, env=env)


def revision(repo, ref="HEAD"):
    return git(repo, "rev-parse", ref).decode().strip()


def clean(repo):
    # Generated experiment records can be committed with the import.
    changes = git(repo, "status", "--porcelain", "--untracked-files=no", "--", ".",
                  ":(exclude)results").decode().strip()
    if changes:
        raise SystemExit(f"Commit the source/tool changes before continuing:\n{changes}")


def write_manifest(workdir, data):
    path = workdir.parent / (workdir.name + ".update.json")
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Update manifest: {path}")


def prepare(args):
    clean(ROOT)
    tracking = json.loads(TRACKING.read_text())
    workdir = args.workdir.resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="allen-update-"))
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(f"Refusing to overwrite {workdir}")
    git(ROOT, "clone", "--no-checkout", str(ROOT), str(workdir))
    for key in ("user.name", "user.email"):
        value = git(ROOT, "config", key).decode().strip()
        git(workdir, "config", key, value)
    git(workdir, "checkout", "-b", "allen/pvfinder-update", tracking["fork_commit"])
    # Capture subsequent edits made in the vendored Allen/ source tree first.
    patch = git(ROOT, "diff", "--binary", tracking["fork_commit"] + "^{tree}",
                "HEAD:Allen", "--", ".", ":(exclude)input")
    if patch:
        git(workdir, "apply", "--index", "-", data=patch)
        git(workdir, "commit", "-m", "Preserve PVFinder edits from inference " + revision(ROOT)[:12])
    git(workdir, "fetch", "--no-tags", URL, args.target)
    target = revision(workdir, "FETCH_HEAD")
    manifest = {"schema": "allen-upstream-update/1", "workdir": str(workdir),
                "inference_before": revision(ROOT), "upstream_url": URL,
                "previous_upstream": tracking["upstream_commit"], "upstream_commit": target}
    try:
        git(workdir, "rebase", "--onto", target, tracking["upstream_commit"])
    except subprocess.CalledProcessError:
        manifest["status"] = "conflicts"
        write_manifest(workdir, manifest)
        raise SystemExit(f"Resolve conflicts in {workdir}, then run git rebase --continue. The inference tree is unchanged.")
    manifest.update(status="prepared", fork_commit=revision(workdir))
    write_manifest(workdir, manifest)
    print(f"Rebased source: {workdir}\nBuild and validate it before importing.")


def import_tree(args):
    clean(ROOT)
    manifest = json.loads(args.manifest.read_text())
    proof = json.loads(args.validation.read_text())
    if proof.get("status") != "PASS" or proof.get("fork_commit") != manifest["fork_commit"]:
        raise SystemExit("Validation must PASS and identify this exact fork commit.")
    if revision(ROOT) != manifest["inference_before"]:
        raise SystemExit("Inference HEAD changed since preparation; prepare the update again.")
    workdir = Path(manifest["workdir"])
    clean(workdir)
    if revision(workdir) != manifest["fork_commit"]:
        raise SystemExit("Prepared source changed since validation.")
    # A temporary index excludes detector/input data from the vendored tree.
    # It never changes either checkout's actual index or local input files.
    with tempfile.TemporaryDirectory(prefix="allen-export-index-") as temp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temp) / "index")}
        git(workdir, "read-tree", "HEAD", env=env)
        inputs = git(workdir, "ls-files", "-z", "--", "input", env=env)
        if inputs:
            git(workdir, "update-index", "--force-remove", "-z", "--stdin", data=inputs, env=env)
        tree = git(workdir, "write-tree", env=env).decode().strip()
    patch = git(workdir, "diff", "--binary", manifest["inference_before"] + ":Allen", tree)
    if patch:
        git(ROOT, "apply", "--check", "--index", "--directory=Allen", "-", data=patch)
    # Preserve upstream ancestry in master as well as the named Allen branch.
    fetch_options = []
    if (git(ROOT, "rev-parse", "--is-shallow-repository").strip() == b"true"
            and git(workdir, "rev-parse", "--is-shallow-repository").strip() == b"false"):
        fetch_options.append("--unshallow")
    git(ROOT, "fetch", *fetch_options, str(workdir), "HEAD")
    git(ROOT, "branch", "-f", "allen/pvfinder", manifest["fork_commit"])
    git(ROOT, "merge", "--no-commit", "--no-ff", "--allow-unrelated-histories", "-s", "ours", "allen/pvfinder")
    if patch:
        git(ROOT, "apply", "--index", "--directory=Allen", "-", data=patch)
    tracking = {"schema": "allen-upstream-tracking/1", "upstream_url": URL,
                "upstream_branch": "master", "upstream_commit": manifest["upstream_commit"],
                "fork_branch": "allen/pvfinder", "fork_commit": manifest["fork_commit"],
                "vendor_tree": tree, "excluded_from_vendor": ["input/"],
                "validation": str(args.validation.resolve().relative_to(ROOT))}
    TRACKING.write_text(json.dumps(tracking, indent=2) + "\n")
    git(ROOT, "add", str(TRACKING.relative_to(ROOT)), tracking["validation"])
    print("Validated source is staged, with upstream ancestry attached. Rebuild, update documentation and commit the merge.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--target", default="master")
    p.add_argument("--workdir", type=Path)
    p.set_defaults(func=prepare)
    p = sub.add_parser("import")
    p.add_argument("manifest", type=Path)
    p.add_argument("--validation", required=True, type=Path)
    p.set_defaults(func=import_tree)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
