#!/usr/bin/python3
###############################################################################
# (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################

import argparse
import json
import sys

from AllenAlgorithmFinder import AllenAlgorithmFinder
from AllenExternLinesGenerator import AllenExternLinesGenerator
from AllenPythonGenerator import AllenPythonGenerator


class AllenCore:
    @staticmethod
    def get_default_properties(default_properties_cmd, prefix_project_folder):
        from subprocess import PIPE, run

        # Run the default_properties executable to get a JSON
        # representation of the default values of all properties of
        # all algorithms
        p = run([default_properties_cmd], stdout=PIPE, encoding="ascii")

        default_properties = None
        if p.returncode == 0:
            default_properties = json.loads(p.stdout)
        else:
            print("Failed to obtain default property values")
            sys.exit(-1)

        return default_properties


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the Allen codebase and generate a python representation of all algorithms."
    )

    parser.add_argument(
        "--filename",
        nargs="?",
        type=str,
        default="algorithms.py",
        help="output filename",
    )
    parser.add_argument(
        "--prefix_project_folder",
        nargs="?",
        type=str,
        default="..",
        help="project location",
    )
    parser.add_argument(
        "--default_properties",
        nargs="?",
        type=str,
        default="",
        help="location of default_properties executable",
    )
    parser.add_argument(
        "--generate",
        nargs="?",
        type=str,
        default="views",
        choices=["views", "extern_lines", "extern_lines_nosepcomp"],
        help="action that will be performed",
    )

    args = parser.parse_args()
    prefix_folder = args.prefix_project_folder + "/"

    if args.generate == "views":
        default_properties = AllenCore.get_default_properties(
            args.default_properties, prefix_folder
        )
        with open("default_properties.json", "w") as f:
            f.write(json.dumps(default_properties, indent=2))

    if args.generate == "views":
        # Generate algorithm python views
        AllenPythonGenerator.write_algorithms_view(default_properties, args.filename)
    elif args.generate == "extern_lines":
        # Write extern lines header file
        all_lines = AllenAlgorithmFinder.find_all_line_instances(prefix_folder)
        AllenExternLinesGenerator.write_extern_lines(all_lines, args.filename, True)
    elif args.generate == "extern_lines_nosepcomp":
        all_lines = AllenAlgorithmFinder.find_all_line_instances(prefix_folder)
        # Write extern lines header file, without separable compilation
        AllenExternLinesGenerator.write_extern_lines(all_lines, args.filename, False)
