from Allen.config import allen_non_event_data_config, run_allen_reconstruction
from GaudiConf.LbExec import Options
from PyConf.application import make_odin
from PyConf.control_flow import CompositeNode, NodeLogic


def main(options: Options):
    def odin_node():
        return CompositeNode("dump_geometry", [make_odin()],
                             combine_logic=NodeLogic.NONLAZY_OR, force_order=True)

    with allen_non_event_data_config.bind(
            dump_geometry=True,
            out_dir="/data/home/melashri/iris/inference/benchmark_results/input_2026_20261006/real_geometry"):
        return run_allen_reconstruction(options, odin_node)
