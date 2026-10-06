import json,hashlib,shutil,struct
from collections import Counter
from pathlib import Path
root=Path('/data/home/melashri/iris/inference');report=root/'results/experiments/v9r1_20261006'
old_checks=json.loads((report/'physics/checks.json').read_text());batch=next(root.glob('benchmark_results/*_v9r1_master_physics'))
def events(path):
 data=path.read_bytes();pos=0;out=[]
 while pos<len(data):
  _,_,n_rec,n_mc=struct.unpack_from('<4I',data,pos)
  end=pos+16+n_rec*9*4+n_mc*4*8
  out.append(hashlib.sha256(data[pos+8:end]).hexdigest());pos=end
 assert pos==len(data)
 return Counter(out)
physics={}
for threshold in ('0.07','0.1'):
 point=batch/f'threshold{threshold}'
 observed=json.loads((point/'compare.json').read_text());old=json.loads((report/f'physics/gpu_work_listtrue_threshold{threshold}/compare.json').read_text())
 assert observed==old,(threshold,'physics metrics differ')
 original=Path(old_checks['batch'])/f'gpu_work_listtrue_threshold{threshold}'
 output_hashes={name:{'main_sha256':hashlib.sha256((point/name).read_bytes()).hexdigest(),'accepted_sha256':hashlib.sha256((original/name).read_bytes()).hexdigest()} for name in ('pvs_pvfinder.bin','pvs_beamline.bin','allen_kde_output.bin','allen_zpeaks.bin')}
 matches={name:events(point/name)==events(original/name) for name in ('pvs_pvfinder.bin','pvs_beamline.bin')}
 assert all(matches.values()),matches
 target=report/f'master_physics/threshold{threshold}';target.mkdir(parents=True,exist_ok=True)
 for name in ('compare.json','validate_peaks.json'):shutil.copy2(point/name,target/name)
 physics[threshold]={'all_physics_metrics_identical':True,'pv_event_payload_multisets_byte_identical':matches,'raw_output_hashes':output_hashes,'in_range':observed['in_range']}
result={'events':10000,'thresholds':physics,'all_physics_metrics_identical':True,'main_artifacts':str(batch.relative_to(root)),'accepted_artifacts':old_checks['batch'],'note':'Raw dump order differs between runs. All 10,000 reconstructed-PV/MC event payloads are byte-identical as multisets for each algorithm and working point, excluding runtime batch/event labels. First-slice KDE/seed dumps cover the first processed slice and are not a matching-event byte comparison here; the fixed 500-event comparison is recorded separately.'}
(report/'master_physics/checks.json').write_text(json.dumps(result,indent=2)+'\n')
shutil.copy2(__file__,report/'master_physics/check_outputs.py')
print('PASS: All physics metrics and 10,000 PV/MC event payloads unchanged at both working points.')
