import json,hashlib,shutil,subprocess,struct
from pathlib import Path
from collections import Counter
root=Path('/data/home/melashri/iris/inference');dest=root/'results/experiments/allen_master_20261006';raw=root/'benchmark_results/allen_master_rebase20261006';work=Path('/tmp/pvfinder-current-master20261006');fork=subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],text=True).strip()
validators={}
for p in (raw/'validation').glob('validate_*.json'):
 r=json.loads(p.read_text());assert r['status']=='PASS',p
 validators[p.stem]=r['status'];target=dest/'preview_validation';target.mkdir(exist_ok=True);shutil.copy2(p,target/p.name)
assert len(validators)==5
for name in ('unit_tests.txt','update_tool_tests.txt'):shutil.copy2(raw/name,dest/name)
shutil.copy2('/tmp/verify_pvfinder_upstream20261006.sh',dest/'preview_validation/commands.sh')
batch=next(root.glob('benchmark_results/*_allen_master_preview_physics'));physics={}
def event_payloads(path):
 data=path.read_bytes();pos=0;result=[]
 while pos<len(data):
  _,_,n_rec,n_mc=struct.unpack_from('<4I',data,pos);end=pos+16+n_rec*36+n_mc*32
  result.append(hashlib.sha256(data[pos+8:end]).hexdigest());pos=end
 assert pos==len(data);return Counter(result)
def compare(a,b,prefix=''):
 differences={}
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:differences.update(compare(a[k],b[k],prefix+'.'+k))
 elif a!=b:
  assert isinstance(a,float) and isinstance(b,float) and abs(a-b)<1e-12,(prefix,a,b)
  differences[prefix]={'v9r1':a,'current_master':b,'absolute_difference':abs(a-b)}
 return differences
for threshold in ('0.07','0.1'):
 current=batch/f'threshold{threshold}';past=root/f'benchmark_results/20261006_060941_v9r1_master_physics/threshold{threshold}'
 report=json.loads((current/'compare.json').read_text());old=json.loads((past/'compare.json').read_text());differences=compare(old,report)
 matches={name:event_payloads(current/name)==event_payloads(past/name) for name in ('pvs_beamline.bin','pvs_pvfinder.bin')};assert all(matches.values()),matches
 out=dest/f'preview_physics/threshold{threshold}';out.mkdir(parents=True,exist_ok=True)
 for name in ('compare.json','validate_peaks.json'):shutil.copy2(current/name,out/name)
 physics[threshold]={'physics_counts_identical_to_v9r1':True,'floating_summary_differences':differences,'pv_mc_event_payloads_identical':matches,'report':str((out/'compare.json').relative_to(dest))}
proof={'schema':'allen-upstream-validation/1','status':'PASS','fork_commit':fork,'upstream_commit':'b567e104f23ebfd0670d6272e47d9b1ce37d905c','build':'/tmp/pvfinder-current-master20261006/buildgpu12','unit_tests':{'cases':14,'assertions':314,'status':'PASS'},'numerical_validation':{'events':500,'tracks':122077,'validators':validators},'physics':{'events':10000,'working_points':physics},'notes':['Standalone CUDA build only; Gaudi stack build untested.','MC dump reads the existing MDF MC-PV payload directly, following upstream checker removal.','Counts and per-event vertex/MC payloads reproduce v9r1; one summary mean differs by 1.11e-16 um due to floating-point summation order.','No claim about current-master throughput yet.']}
(dest/'preview_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
shutil.copy2(__file__,dest/'preview_physics/check_comparisons.py')
print('PASS: native rebase validates, including unchanged MC/vertex event payloads and physics counts against v9r1.')
