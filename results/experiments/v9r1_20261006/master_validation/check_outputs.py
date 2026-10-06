import json,hashlib
from pathlib import Path
import numpy as np
root=Path('/data/home/melashri/iris/inference')
current=root/'benchmark_results/v9r1_master_transfer20261006/validation'
old=root/'local_scripts/rebase_v9r1/optimization/validation/gpu_work_list'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=sorted({p.name for p in current.glob('*.bin')} & {p.name for p in old.glob('*.bin')})
matches={name:digest(current/name)==digest(old/name) for name in files}
required=('allen_fc_csr.bin','allen_fc_interval_features.bin','allen_fc_histogram.bin','allen_kde_output.bin','allen_ncw_input.bin','allen_zpeaks.bin','pvs_beamline.bin','pvs_pvfinder.bin')
assert all(matches[name] for name in required)
def load(folder,name,width):
 return np.fromfile(folder/name,dtype=np.uint32,offset=16).reshape(-1,width)
new_rows=np.ascontiguousarray(np.concatenate([load(current,'allen_fc_track_states.bin',6),load(current,'allen_fc_track_features.bin',9)],axis=1)).view('V60').ravel()
old_rows=np.ascontiguousarray(np.concatenate([load(old,'allen_fc_track_states.bin',6),load(old,'allen_fc_track_features.bin',9)],axis=1)).view('V60').ravel()
offsets=load(current,'allen_fc_track_offsets.bin',1).ravel().tolist()
if len(offsets)==500:offsets.append(len(new_rows))
assert len(offsets)==501
tracks_equal=all(np.array_equal(np.sort(new_rows[start:end]),np.sort(old_rows[start:end])) for start,end in zip(offsets[:-1],offsets[1:]))
assert tracks_equal
report={'byte_equal':matches,'required_output_files':list(required),'required_outputs_byte_identical':True,'track_state_feature_multisets_equal_per_event':tracks_equal,'note':'Raw track packing order differs; joint state/feature rows are byte-identical as multisets within each event. CSR counts and FC/UNet/peak/fitted-PV outputs are byte-identical.'}
path=root/'results/experiments/v9r1_20261006/master_validation';path.mkdir(exist_ok=True)
(path/'byte_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
