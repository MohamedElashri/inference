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

from AllenGeneratorUtils import get_namespace


class AllenPythonGenerator:
    """Generate python representations of Allen Algorithms"""

    @staticmethod
    def write_algorithm_code(algorithm):
        namespace, _ = get_namespace(algorithm["type"])
        _, name = get_namespace(algorithm["name"])
        s = f"class {name}(AllenAlgorithm):\n"

        # Slots
        s += "  __slots__ = OrderedDict(\n"
        for param in algorithm["parameters"]:
            dependencies = (
                [
                    '"' + dep.replace(namespace + "::Parameters::", "") + '"'
                    for dep in param["dependencies"]
                ]
                if "dependencies" in param
                else []
            )
            dependencies = "[" + ", ".join(dependencies) + "]" if dependencies else "[]"
            _, typename = get_namespace(param["typename"])
            io = "R" if "input" in param["kind"].lower() else "W"
            s += f"    {typename} = AllenDataHandle('{param['scope']}', {dependencies}, '{typename}', '{io}', '{param['type']}'),\n"

        # Properties
        for pn, [dv, data_type, descr] in algorithm["properties"].items():
            # Quotes have to be added for properties that hold a string
            if type(dv) is str:
                dv = f'"{dv}"'

            if data_type == "dim3" or data_type.startswith("std::array"):
                dv = tuple(dv)

            # Write the code for the property
            s += f"    {pn} = {dv}, # ({data_type}) {descr}\n"
        s = s[:-1]
        s += "\n  )\n"

        # aggregates: parameters marked optional or aggregate
        s += "  aggregates = ("
        for param in algorithm["parameters"]:
            if "aggregate" in param or "optional" in param:
                _, typename = get_namespace(param["typename"])
                s += f"\n    '{typename}',"
        s += ")\n\n"
        s += "  @staticmethod\n"
        s += "  def category():\n"
        s += f"    return '{algorithm['scope']}'\n\n"
        s += "  def __new__(self, name, **kwargs):\n"
        s += "    instance = AllenAlgorithm.__new__(self, name)\n"
        s += "    for n,v in kwargs.items():\n"
        s += "      setattr(instance, n, v)\n"
        s += "    return instance\n\n"
        s += "  @classmethod\n"
        s += "  def getType(cls):\n"
        s += f"    return '{name}'\n\n"
        s += "  @classmethod\n"
        s += "  def getName(cls):\n"
        s += f"    return '{algorithm['name']}'\n\n\n"

        return s

    @staticmethod
    def write_algorithms_view(algorithms, filename):
        s = "from AllenCore.AllenKernel import AllenAlgorithm, AllenDataHandle\n"
        s += "from collections import OrderedDict\n\n"
        for name, algorithm in algorithms.items():
            s += AllenPythonGenerator.write_algorithm_code(algorithm)
        with open(filename, "w") as f:
            f.write(s)
