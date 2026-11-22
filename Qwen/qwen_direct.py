"""Qwen3.5-4B zero-shot test evaluation via local Ollama; no training."""
import argparse,base64,csv,hashlib,io,json,os,re,sqlite3,time,sys
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.registry import TASKS
from src.data import image_references,resolve_image
MODEL='qwen3.5:4b'
PROMPT_CONFIG=json.loads((Path(__file__).with_name('prompts.json')).read_text(encoding='utf-8'))
PROMPTS=PROMPT_CONFIG['tasks']

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def dump(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)
def parse_label(text,codes):
 text=re.sub(r'<think>.*?</think>','',text,flags=re.S).replace('**','').strip().strip('`').strip()
 m=re.fullmatch(r'(?:标签\s*[:：]\s*)?(\d+)[。.]?',text)
 if m and int(m[1]) in codes:return int(m[1])
 explicit=re.findall(r'(?m)^[ \t#]*(?:最终(?:的)?\s*)?(?:标签|答案|分类结果|判断结果|结论)\s*[:：]\s*(\d+)[。.]?[ \t]*$',text)
 valid={int(v) for v in explicit if int(v) in codes}
 if explicit and (len(valid)!=1 or any(int(v) not in codes for v in explicit)):raise ValueError('Conflicting or invalid final labels')
 if len(valid)==1:return valid.pop()
 last=text.splitlines()[-1].strip() if text else ''
 if re.fullmatch(r'\d+[。.]?',last) and int(last.rstrip('。.')) in codes:return int(last.rstrip('。.'))
 raise ValueError('Expected one valid label, got '+repr(text[:120]))
def messages_for(row,task,image):
 message={'role':'user','content':PROMPTS[task]+PROMPT_CONFIG['news_prefix']+row['content']}
 if task!='text_binary':
  if image is not None:message['images']=[image]
  else:message['content']+=PROMPT_CONFIG['missing_image_suffix']
 return [message]

class OllamaClient:
 def __init__(self,url='http://127.0.0.1:11434',timeout=600,temperature=0,num_ctx=16384,num_predict=8192):
  p=urlparse(url)
  if p.scheme!='http' or p.hostname not in ['127.0.0.1','localhost','::1']:raise ValueError('Use a local Ollama loopback URL')
  self.url=url.rstrip('/');self.timeout=timeout;self.digest=None;self.temperature=temperature;self.num_ctx=num_ctx;self.num_predict=num_predict
 def request(self,endpoint,payload=None):
  req=Request(self.url+endpoint,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
  with urlopen(req,timeout=self.timeout) as response:return json.load(response)
 def preflight(self,vision=True):
  models=self.request('/api/tags').get('models',[]);found=next((m for m in models if MODEL in [m.get('name'),m.get('model')]),None)
  if found is None:raise RuntimeError('Missing local model. Run: ollama pull '+MODEL)
  info=self.request('/api/show',{'model':MODEL})
  if vision and 'vision' not in info.get('capabilities',[]):raise RuntimeError('Installed Qwen model does not report vision capability')
  self.digest=found.get('digest') or digest(info);return self.digest
 def predict(self,messages,seed):
  payload={'model':MODEL,'messages':messages,'stream':False,'think':False,'options':{'num_ctx':self.num_ctx,'num_predict':self.num_predict,'temperature':self.temperature,'seed':seed}}
  if getattr(self,'constrained_output',False):payload['format']={'type':'integer','enum':self.label_codes}
  d=self.request('/api/chat',payload)
  if d.get('error') or not d.get('done') or d.get('done_reason')=='length':raise RuntimeError('Incomplete Ollama answer: '+str(d.get('error') or d.get('done_reason')))
  meta={k:d.get(k) for k in ['model','created_at','prompt_eval_count','eval_count']};meta['output_format']='integer_enum' if 'format' in payload else 'plain_text'
  return d.get('message',{}).get('content',''),meta

def image_input(root,row):
 from PIL import Image,ImageFile
 for ref in image_references(row):
  path=resolve_image(root/'images',ref)
  if not path.is_file():raise FileNotFoundError(path)
  old=ImageFile.LOAD_TRUNCATED_IMAGES
  try:
   if ref=='7557/tampered.jpg':ImageFile.LOAD_TRUNCATED_IMAGES=True
   with Image.open(path) as image:
    data=io.BytesIO();image.convert('RGB').save(data,format='JPEG',quality=95)
   return base64.b64encode(data.getvalue()).decode(),ref
  finally:ImageFile.LOAD_TRUNCATED_IMAGES=old
 return None,None

def acquire_lock(path):
 lock=sqlite3.connect(str(path),timeout=0)
 try:lock.execute('BEGIN EXCLUSIVE')
 except sqlite3.OperationalError:
  lock.close();raise RuntimeError('Another Qwen process is already working in this output directory')
 return lock

def evaluate_direct(root,task,client,sleep=time.sleep):
 from src.metrics import metrics
 root=Path(root);out=root/'results'/task/'qwen35_4b'/'seed_2026';out.mkdir(parents=True,exist_ok=True)
 rows=json.loads((root/'data/test.json').read_text(encoding='utf-8'));field,codes=TASKS[task]
 if len({str(r['id']) for r in rows})!=len(rows):raise ValueError('Duplicate sample IDs')
 inventory=[]
 if task!='text_binary':
  for ref in sorted({ref for r in rows for ref in image_references(r)}):
   path=resolve_image(root/'images',ref);st=path.stat();inventory.append([ref,st.st_size,st.st_mtime_ns])
 contract={'model':MODEL,'model_digest':client.digest,'task':task,'test_sha256':hashlib.sha256((root/'data/test.json').read_bytes()).hexdigest(),'image_inventory':digest(inventory),'prompt':PROMPTS[task],'seed':2026,'temperature':getattr(client,'temperature',0),'think':False,'num_ctx':getattr(client,'num_ctx',16384),'num_predict':getattr(client,'num_predict',8192),'image_policy':'first readable image, RGB JPEG95, original resolution; missing preserved','version':1}
 lock=acquire_lock(out/'run_lock.sqlite3')
 db=sqlite3.connect(str(out/'requests.sqlite3'))
 try:
  db.execute('CREATE TABLE IF NOT EXISTS metadata (signature TEXT)');db.execute('CREATE TABLE IF NOT EXISTS answers (id TEXT PRIMARY KEY, answer TEXT)')
  old=db.execute('SELECT signature FROM metadata').fetchone();signature=digest(contract)
  if old and old[0]!=signature:raise ValueError('Qwen cache belongs to different data/model/settings; archive this Qwen output directory before starting a new configuration')
  if not old:db.execute('INSERT INTO metadata VALUES (?)',(signature,));db.commit()
  config={**contract,'format_recovery':'explicit final-label parsing; integer-enum constrained retry after invalid response','batch_size':1,'inference_only':True,'class_probabilities':'not returned; one-hot hard decisions used for metric calculation'};dump(out/'config.json',config)
  dump(out/'status.json',{'status':'running','test_sha256':contract['test_sha256']})
  result=[]
  for i,row in enumerate(rows):
   cached=db.execute('SELECT answer FROM answers WHERE id=?',(str(row['id']),)).fetchone()
   if cached:answer=json.loads(cached[0]);source='cached'
   else:
    image,reference=image_input(root,row) if task!='text_binary' else (None,None)
    attempts=0;client.label_codes=codes;client.constrained_output=False
    while True:
     seed=2026 if attempts==0 else int(digest([2026,str(row['id']),task,attempts])[:8],16)%(2**31)
     raw=None
     try:
      raw,meta=client.predict(messages_for(row,task,image),seed);label=parse_label(raw,codes)
      answer={'prediction':label,'raw_response':raw,'request_seed':seed,'retry_attempt':attempts,'image_reference':reference,'metadata':meta}
      db.execute('INSERT OR REPLACE INTO answers VALUES (?,?)',(str(row['id']),json.dumps(answer,ensure_ascii=False)));db.commit();break
     except Exception as exc:
      client.constrained_output=True
      with (out/'failed_responses.jsonl').open('a',encoding='utf-8') as failure_log:
       failure_log.write(json.dumps({'id':str(row['id']),'attempt':attempts,'error':str(exc),'raw_response':raw if 'raw' in locals() else None},ensure_ascii=False)+'\n')
      delay=min(2**min(attempts,5),30);attempts+=1;print(f'Qwen {task} id={row["id"]} retry={attempts} wait={delay}s: {exc}',flush=True);sleep(delay)
    source='generated'
   result.append({'id':str(row['id']),'label':int(row[field]),**answer})
   print(f'Qwen {task} {i+1}/{len(rows)} id={row["id"]} {source}',flush=True)
  y=[codes.index(r['label']) for r in result];prob=[[int(c==r['prediction']) for c in codes] for r in result];report=metrics(y,prob,codes)
  report['prediction_kind']='direct hard labels; no probability calibration/AUC';report['input_mode']='text' if task=='text_binary' else 'text+available image'
  if task=='text_binary':report['binary_field_semantics'].update(real='text unmodified',fake='text modified')
  dump(out/'test_metrics.json',report)
  tmp=out/'test_predictions.jsonl.tmp'
  with tmp.open('w',encoding='utf-8') as f:
   for record,hard in zip(result,prob):f.write(json.dumps({**record,'probabilities':hard,'probabilities_kind':'one_hot_hard_prediction'},ensure_ascii=False)+'\n')
  os.replace(tmp,out/'test_predictions.jsonl')
  tmp=out/'test_predictions.csv.tmp'
  with tmp.open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=['id','label','prediction','correct','image_reference','raw_response']);writer.writeheader()
   for record in result:
    exported={k:record.get(k) for k in ['id','label','prediction','image_reference','raw_response']};exported['correct']=record['label']==record['prediction'];writer.writerow(exported)
  os.replace(tmp,out/'test_predictions.csv');dump(out/'status.json',{'status':'complete','test_macro_f1':report['macro_f1'],'test_sha256':contract['test_sha256']})
  print('Qwen complete:',out,flush=True)
 finally:db.close();lock.rollback();lock.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--ollama-url',default='http://127.0.0.1:11434');p.add_argument('--temperature',type=float,default=0);p.add_argument('--num-ctx',type=int,default=16384);p.add_argument('--num-predict',type=int,default=8192);p.add_argument('--timeout',type=float,default=600);p.add_argument("--plan",action="store_true");args=p.parse_args()
 if args.plan:print(json.dumps({**vars(args),"root":str(args.root),"model":MODEL,"prompt":PROMPTS[args.task]},ensure_ascii=False,indent=2));return
 client=OllamaClient(args.ollama_url,args.timeout,args.temperature,args.num_ctx,args.num_predict);client.preflight(vision=args.task!='text_binary');evaluate_direct(args.root,args.task,client)
if __name__=='__main__':main()
