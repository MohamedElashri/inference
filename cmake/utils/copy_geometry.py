#!/usr/bin/env python3
###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "COPYING".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
"""
copy_geometry.py

Copy a geometry directory to the build tree, preserving symlinks,
and rewiring broken magfield symlinks to local generated .bin files.

This is to be called by the cmake, do not run this yourself
"""

import os
import sys
import shutil
import re

src_dir = sys.argv[1]
dst_dir = sys.argv[2]
local_magfields_dir = sys.argv[3]

# Ensure destination exists
os.makedirs(dst_dir, exist_ok=True)
os.makedirs(local_magfields_dir, exist_ok=True)

# Walk source tree
for root, dirs, files in os.walk(src_dir):
    rel_root = os.path.relpath(root, src_dir)
    dest_root = os.path.join(dst_dir, rel_root) if rel_root != "." else dst_dir

    os.makedirs(dest_root, exist_ok=True)

    # Create directories
    for d in dirs:
        os.makedirs(os.path.join(dest_root, d), exist_ok=True)

    # Copy files and handle symlinks
    for f in files:
        src_f = os.path.join(root, f)
        dst_f = os.path.join(dest_root, f)

        if os.path.islink(src_f):
            linkto = os.readlink(src_f)
            # Remove existing dst symlink if present
            if os.path.lexists(dst_f):
                os.remove(dst_f)
            os.symlink(linkto, dst_f)

            # Rewire broken magfield symlinks
            if not os.path.exists(os.path.join(root, linkto)):
                base = os.path.basename(linkto)
                m = re.match(r'field\.([^.]+)\.(up|down)\.bin', base)
                if m:
                    version, polarity = m.groups()
                    local_target = os.path.join(
                        local_magfields_dir,
                        f'magfield.{version}.{polarity}.bin')
                    if os.path.exists(local_target):
                        os.remove(dst_f)
                        os.symlink(local_target, dst_f)

        else:
            # Regular file: copy
            shutil.copy2(src_f, dst_f)
