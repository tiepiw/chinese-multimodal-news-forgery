#!/usr/bin/env python3
"""Unified offline baseline entry point. No weight downloads are performed."""
import os,sys,json,argparse,subprocess,hashlib,importlib.util
from datetime import datetime
import uuid
from pathlib import Path
from src.profiles import resolve_profile
from src.registry import DEFAULT_MODELS,TASKS,experiments
from src.data import load_splits,audit_records,image_references,resolve_image
ROOT=Path(__file__).resolve().parent
WEIGHTS={'bert':['pretrained/chinese-bert-wwm-ext/config.json','pretrained/chinese-bert-wwm-ext/vocab.txt'], 'resnet18':['pretrained/resnet18/resnet18-f37072fd.pth'],'vgg19':['pretrained/vgg19/vgg19-dcbb9e9d.pth'],'deit':['pretrained/deit/deit_base_patch16_224-b5f2ef4d.pth']}
def required_weights(models):
 kinds=set()
 if any(m in ['bert','concat','cafe','hammer','arg','himf_moe'] for m in models):kinds.add('bert')
 if any(m in ['concat','cafe','himf_moe'] for m in models):kinds.add('resnet18')
 if 'eann' in models:kinds.add('vgg19')
 if 'hammer' in models:kinds.add('deit')
 missing=[f for k in kinds for f in WEIGHTS[k] if not (ROOT/f).is_file()]
 if 'bert' in kinds:
  p=ROOT/'pretrained/chinese-bert-wwm-ext'
  if not any((p/name).is_file() for name in ['model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json']):missing.append('pretrained/chinese-bert-wwm-ext/weights (model.safetensors or pytorch_model.bin)')
 return missing

def check(models,verify_hashes=False,ollama_url="http://127.0.0.1:11434"):
 dependencies=['numpy','sklearn','PIL'] if set(models)=={'qwen35_4b'} else ['torch','torchvision','transformers','timm','numpy','sklearn','PIL']
 data=load_splits(ROOT/'data');report={'data':audit_records(data),'missing_weights':required_weights(models),'missing_dependencies':[m for m in dependencies if importlib.util.find_spec(m) is None]}
 refs={ref for rows in data.values() for row in rows for ref in image_references(row)}
 report['missing_ollama']=[]
 if 'qwen35_4b' in models:
  from Qwen.qwen_direct import OllamaClient
  try:OllamaClient(ollama_url).preflight()
  except Exception as exc:report['missing_ollama'].append(str(exc))
 report['missing_arg_analyses']=[]
 if 'arg' in models:
  from src.arg_data import load_arg_rows
  for s,rows in data.items():
   try:load_arg_rows(ROOT/'data'/f'{s}.json',rows)
   except (ValueError,FileNotFoundError) as exc:report['missing_arg_analyses'].append(str(exc))
 report['image_references']=len(refs);report['missing_images']=[ref for ref in sorted(refs) if not resolve_image(ROOT/'images',ref).is_file()]
 if verify_hashes:
  hashes=json.loads((ROOT/'data/manifest.json').read_text())['images']['sha256']
  report['image_hash_mismatches']=[ref for ref in sorted(refs) if resolve_image(ROOT/'images',ref).is_file() and hashlib.sha256(resolve_image(ROOT/'images',ref).read_bytes()).hexdigest()!=hashes[ref]]
 report['known_truncated_image']='7557/tampered.jpg: original bytes preserved; explicitly tolerant PIL decoding'
 return report

def job_state(out,model):
 status=out/'status.json'
 required=['test_metrics.json','test_predictions.jsonl','test_predictions.csv']
 try:done=json.loads(status.read_text(encoding='utf-8')).get('status')=='complete'
 except (OSError,ValueError):done=False
 if done and all((out/f).is_file() and (out/f).stat().st_size for f in required):
  if model=='qwen35_4b':
   try:
    config=json.loads((out/'config.json').read_text(encoding='utf-8'));source=out.parents[3]/'data/test.json'
    if config.get('test_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():return 'resume'
   except (OSError,ValueError):return 'resume'
  return 'skip'
 if model=='himf_moe' and (out/'last.pt').is_file():return 'resume'
 if model=='qwen35_4b' and out.exists() and any(out.iterdir()):return 'resume'
 if out.exists() and any(out.iterdir()):return 'restart_archived'
 return 'train'

def archive_output(out):
 destination=out.with_name(out.name+'_archive_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
 out.rename(destination);return destination

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--models',nargs='+',default=DEFAULT_MODELS);ap.add_argument('--tasks',nargs='+',default=list(TASKS));ap.add_argument('--check',action='store_true');ap.add_argument('--verify-hashes',action='store_true');ap.add_argument('--plan',action='store_true');ap.add_argument('--epochs',type=int,default=None);ap.add_argument('--batch-size',type=int,default=None);ap.add_argument('--max-length',type=int,default=None);ap.add_argument('--device',default='auto');ap.add_argument('--patience',type=int,default=None);ap.add_argument('--seed',type=int,default=None);ap.add_argument('--lr',type=float,default=None);ap.add_argument('--bert-lr',type=float,default=None);ap.add_argument('--vision-lr',type=float,default=None);ap.add_argument('--weight-decay',type=float,default=None);ap.add_argument('--warmup-epochs',type=int,default=None);ap.add_argument('--optimizer',default=None);ap.add_argument('--schedule',default=None);ap.add_argument('--ollama-url',default='http://127.0.0.1:11434');ap.add_argument('--overwrite',action='store_true');ap.add_argument('--continue-on-error',action='store_true');ap.add_argument('--resume',action='store_true',help='Compatibility flag; automatic skip/resume is now the default');args=ap.parse_args()
 seed=args.seed if args.seed is not None else 2026
 if 'qwen35_4b' in args.models and seed!=2026:raise ValueError('Qwen direct evaluation uses inference seed 2026; use its separate shell entry point for the fixed protocol')
 jobs=experiments(args.models,args.tasks)
 print('Experiment plan:',', '.join(m+'/'+t for m,t in jobs),flush=True)
 states={(m,t):job_state(ROOT/'results'/t/m/f'seed_{seed}',m) for m,t in jobs}
 for m,t in jobs:print(m,t,':',states[m,t],flush=True)
 if args.plan:return
 pending=[(m,t) for m,t in jobs if args.overwrite or states[m,t]!='skip']
 if not pending:
  print('All selected experiments are complete; skipped.');return
 report=check(sorted({m for m,t in pending}),args.verify_hashes,args.ollama_url)
 print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
 blocked=any(report[k] for k in ['missing_weights','missing_dependencies','missing_images','missing_arg_analyses','missing_ollama']) or report.get('image_hash_mismatches')
 if blocked:
  print('Preflight failed. Resolve the missing dependencies, checkpoints, or Ollama model listed above. No experiments started.',flush=True);sys.exit(2)
 if args.check:print('Preflight passed.');return
 env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONHASHSEED':str(seed)};runs=[]
 for m,t in jobs:
  out=ROOT/'results'/t/m/f'seed_{seed}'
  state=states[m,t]
  if state=='skip' and not args.overwrite:
   print('Skipping completed:',m,t,flush=True)
   runs.append({'model':m,'task':t,'status':'skipped_completed'});continue
  if args.overwrite or state=='restart_archived':
   if out.exists() and any(out.iterdir()):print('Archived previous output:',archive_output(out),flush=True)
   if state=='restart_archived' and not args.overwrite:print('No full training state is available; archived the old output and restarting:',m,t,flush=True)
  if m=='qwen35_4b':
   cmd=[sys.executable,'-m','Qwen.qwen_direct','--root',str(ROOT),'--task',t,'--ollama-url',args.ollama_url]
  elif m=='himf_moe':
   cmd=[sys.executable,str(ROOT/'HIMF_MoE/train.py'),'--device',args.device,'--output-dir',str(out)]
  else:
   cmd=[sys.executable,'-m','src.train','--root',str(ROOT),'--model',m,'--task',t,'--device',args.device,'--seed',str(seed)]
  for flag in ['epochs','batch_size','max_length','patience','seed','lr','bert_lr','vision_lr','weight_decay','warmup_epochs','optimizer','schedule']:
   if m=='qwen35_4b':continue
   if m=='himf_moe' and flag in ['lr','vision_lr','warmup_epochs','optimizer','schedule']:continue
   if getattr(args,flag) is not None:cmd+=['--'+flag.replace('_','-'),str(getattr(args,flag))]
  out.parent.mkdir(parents=True,exist_ok=True);log=out.parent/f'{m}_seed{seed}_console.log'
  print('Starting',m,t,flush=True)
  with log.open('a',encoding='utf-8') as f:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
   for line in child.stdout:print(line,end='',flush=True);f.write(line);f.flush()
   code=child.wait()
  runs.append({'model':m,'task':t,'return_code':code,'status':'complete' if code==0 else 'failed','log':str(log.relative_to(ROOT))})
  (ROOT/'results/run_manifest.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2)+'\n')
  if code and not args.continue_on_error:sys.exit(code)
 if any(r.get('return_code',0) for r in runs):sys.exit(1)
 print('All selected experiments completed. Outputs are in results/.')
if __name__=='__main__':main()
