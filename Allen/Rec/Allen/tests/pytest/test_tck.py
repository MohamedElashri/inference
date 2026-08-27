###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
import subprocess

from Allen.tck import clone_metainfo_repository


def git(repository, *arguments):
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_clone_metainfo_repository_uses_master_when_source_head_is_transient(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.name", "Allen test")
    git(source, "config", "user.email", "allen-test@invalid")
    (source / "metadata.json").write_text("{}\n")
    git(source, "add", "metadata.json")
    git(source, "commit", "-q", "-m", "Initialize metadata repository")
    git(source, "branch", "-M", "master")
    git(source, "switch", "-q", "-c", "key-deadbeef")

    destination = tmp_path / "destination"
    clone_metainfo_repository(source, destination)

    assert git(source, "branch", "--show-current") == "key-deadbeef"
    assert git(destination, "branch", "--show-current") == "master"
    assert git(destination, "rev-parse", "--verify", "master")
