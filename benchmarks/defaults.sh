# Shared defaults for the build, benchmark and validation tools.
PVF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PVF_BUILD_DIR=${ALLEN_BUILD_DIR:-$PVF_REPO_ROOT/Allen/build}
PVF_MODEL=unet16_lc4_scnone_asym5_best_bf16
PVF_DEVICE=${DEVICE:-2}
PVF_EVENTS=500
PVF_MEMORY=500
PVF_REPETITIONS=1000
PVF_THREADS=16
PVF_REPEATS=5
PVF_FC_GRID_FRACTION=0.0625
PVF_UNET_GRID_FRACTION=0.125
export LD_LIBRARY_PATH="${CUDNN_ROOT:-$HOME/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
# Input and its matching dumped geometry are selected together for every tool.
PVF_MDF=${MDF_FILE:-$PVF_REPO_ROOT/Allen/input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf}
PVF_GEOMETRY=${GEOMETRY_DIR:-$PVF_REPO_ROOT/Allen/input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry}
