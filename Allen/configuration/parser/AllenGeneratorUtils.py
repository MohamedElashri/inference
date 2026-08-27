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


def get_namespace(demangled_type):
    type_str = demangled_type.strip()

    # Find the last top-level "::" that's not inside template brackets
    bracket_level = 0
    last_namespace_pos = -1

    for i in range(len(type_str) - 1, -1, -1):
        char = type_str[i]
        if char == ">":
            bracket_level += 1
        elif char == "<":
            bracket_level -= 1
        elif char == ":" and bracket_level == 0:
            if i > 0 and type_str[i - 1] == ":":
                # Found a :: at top level
                last_namespace_pos = i - 1
                break

    if last_namespace_pos != -1:
        namespace = type_str[:last_namespace_pos].strip()
        name = type_str[last_namespace_pos + 2 :].strip()
        name = name.split("<")[0].strip() if "<" in name else name
        return namespace, name

    # No namespace found, just return the base name
    base_name = type_str.split("<")[0].strip()
    return "", base_name
