"""One independent run. Select only with validation macro F1, test once afterward."""
import argparse,json,random,hashlib,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Sampler
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
from .data import load_splits,audit_records
from .runtime_data import NewsDataset,ARGDataset,build_vocab
from .arg_data import load_arg_rows
from .registry import TASKS
from .profiles import resolve_profile
from .training_policy import next_stale
from .factory import build_model
from .metrics import metrics
from tqdm.auto import tqdm

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
class TrainingBatches(Sampler):
 """Keep every sample, merge a singleton tail for CAFE BatchNorm."""
 def __init__(self,n,batch,seed):self.n=n;self.batch=batch;self.rng=random.Random(seed)
 def __iter__(self):
  ids=list(range(self.n));self.rng.shuffle(ids)
  # Advance one generator initialized exactly with the experiment seed.
  batches=[ids[i:i+self.batch] for i in range(0,self.n,self.batch)]
  if len(batches)>1 and len(batches[-1])==1:
   tail=batches.pop();batches[-1].extend(tail)
  yield from batches
 def __len__(self):
  n=(self.n+self.batch-1)//self.batch
  return n-1 if self.n%self.batch==1 and n>1 else n

def fit_events(data,out,seed):
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.cluster import MiniBatchKMeans
 import pickle
 vec=TfidfVectorizer(analyzer='char',ngram_range=(1,2),max_features=5000)
 x=vec.fit_transform([r['content'] for r in data['train']]);k=min(20,x.shape[0]);km=MiniBatchKMeans(n_clusters=k,random_state=seed,n_init=10,batch_size=256).fit(x)
 result={s:km.predict(vec.transform([r['content'] for r in rows])).tolist() for s,rows in data.items()}
 with (out/'pseudo_events.pkl').open('wb') as f:pickle.dump({'vectorizer':vec,'kmeans':km},f)
 dump(out/'pseudo_events.json',{'fit_split':'train','event_count':k,'seed':seed,'assignment_counts':{s:dict(Counter(e)) for s,e in result.items()}})
 return result,k

def evaluate(model,loader,device,codes):
 model.eval();y=[];probs=[];ids=[]
 with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []),torch.no_grad():
  torch.manual_seed(2026)
  for batch in loader:
   ids.extend(batch['id']);b={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()};o=model(b);p=o['logits'].softmax(-1)
   if not torch.isfinite(p).all():raise RuntimeError('Non-finite evaluation probabilities')
   y+=b['target'].cpu().tolist();probs+=p.cpu().tolist()
 return metrics(y,probs,codes),ids,y,probs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--task',choices=TASKS,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--epochs',type=int,default=None);ap.add_argument('--batch-size',type=int,default=None);ap.add_argument('--max-length',type=int,default=None);ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--device',default='auto');ap.add_argument('--patience',type=int,default=None);ap.add_argument('--overwrite',action='store_true');ap.add_argument('--lr',type=float,default=None);ap.add_argument('--bert-lr',type=float,default=None);ap.add_argument('--vision-lr',type=float,default=None);ap.add_argument('--weight-decay',type=float,default=None);ap.add_argument('--warmup-epochs',type=int,default=None);ap.add_argument('--optimizer',choices=['Adam', 'AdamW', 'Adadelta'],default=None);ap.add_argument('--schedule',choices=['constant', 'linear', 'cosine'],default=None);args=ap.parse_args()
 profile=resolve_profile(args.model,vars(args),load_splits(args.root/'data')['train'])
 for key in ['epochs','batch_size','max_length','patience']:setattr(args,key,profile[key])
 print('Effective training parameters:',json.dumps(profile,ensure_ascii=False),flush=True)
 if args.batch_size<2:raise ValueError('Use batch size >=2 (CAFE BatchNorm)')
 root=args.root.resolve();out=root/'results'/args.task/args.model/f'seed_{args.seed}'
 if out.exists() and any(out.iterdir()) and not args.overwrite:raise FileExistsError('Output already exists; use --overwrite explicitly: '+str(out))
 out.mkdir(parents=True,exist_ok=True);dump(out/'status.json',{'status':'running'});seed_all(args.seed)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
 data=load_splits(root/'data');audit=audit_records(data);field,codes=TASKS[args.task]
 if args.model=='arg':
  if args.task!='text_binary':raise ValueError('ARG only supports text_binary')
  data={s:load_arg_rows(root/'data'/f'{s}.json',rows) for s,rows in data.items()}
 vocab=build_vocab(data['train']);dump(out/'vocab.json',vocab)
 need_bert=args.model not in ['textcnn','eann','image_only'];tokenizer=AutoTokenizer.from_pretrained(str(root/'pretrained/chinese-bert-wwm-ext'),local_files_only=True,use_fast=False) if need_bert else None
 events,k=fit_events(data,out,args.seed) if args.model=='eann' else (None,20)
 dataset_class=ARGDataset if args.model=='arg' else NewsDataset
 datasets={s:dataset_class(rows,args.task,root/'images',tokenizer=tokenizer,vocab=vocab,max_length=args.max_length,image_size=256 if args.model=='hammer' else 224,use_image=args.model not in ['bert','textcnn','arg'],events=events[s] if events else None) for s,rows in data.items()}
 loaders={s:DataLoader(ds,batch_sampler=TrainingBatches(len(ds),args.batch_size,args.seed),num_workers=0) if s=='train' else DataLoader(ds,batch_size=args.batch_size,num_workers=0,shuffle=False) for s,ds in datasets.items()}
 model=build_model(args.model,len(codes),root,len(vocab),k).to(device)
 params={'bert':[],'vision':[],'head':[]}
 for name,p in model.named_parameters():
  if not p.requires_grad:continue
  group='bert' if name.startswith(('text.','bert.','core.bert_content.','core.bert_FTR.')) else 'vision' if name.startswith(('vision.','vgg.')) else 'head';params[group].append(p)
 rates={g:profile.get(g+'_lr',profile['lr']) for g in params}
 groups=[{'params':v,'lr':rates[g],'name':g} for g,v in params.items() if v]
 optimizer_class=getattr(torch.optim,profile['optimizer'])
 kwargs={'weight_decay':profile['weight_decay']}
 if profile['optimizer']=='Adadelta':kwargs.update(rho=.95,eps=1e-6)
 opt=optimizer_class(groups,**kwargs)
 total=len(loaders['train'])*args.epochs
 if profile['schedule']=='linear':scheduler=get_linear_schedule_with_warmup(opt,int(.1*total),total)
 elif profile['schedule']=='cosine':
  import math
  warmup=profile['warmup_epochs']*len(loaders['train'])
  def lr_factor(base):
   floor=1e-6/base
   return lambda step: floor+(1-floor)*step/warmup if step<warmup else floor+(1-floor)*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))
  scheduler=torch.optim.lr_scheduler.LambdaLR(opt,[lr_factor(g['lr']) for g in opt.param_groups])
 else:scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
 counts=Counter(int(r[field]) for r in data['train']);weight=torch.tensor([len(data['train'])/(len(codes)*counts[c]) for c in codes],device=device)
 criterion=nn.CrossEntropyLoss()
 config={**vars(args),'root':str(root),'device':str(device),'selection_metric':'validation macro_f1','input_fields':(['content']+(['images'] if args.model not in ['bert','textcnn','arg'] else [])),'class_weights':None,'parameter_profile':profile,'optimizer':profile['optimizer'],'weight_decay':profile['weight_decay'],'scheduler':profile['schedule'],'optimizer_lr':rates,'data_audit':audit,'dataset_sha256':{s:hashlib.sha256((root/'data'/f'{s}.json').read_bytes()).hexdigest() for s in data},'initialization':'local pretrained only; task/fusion heads random','epochs_actual':0,'tolerant_image_decoding':['7557/tampered.jpg'],'versions':{'torch':torch.__version__}}
 if args.model=='arg':
  config['input_fields']=['content','LLM-generated td_rationale','LLM-generated cs_rationale'];config['label_semantics']={'0':'text unmodified','1':'text modified'};config['arg_auxiliary_loss_weights']={'judgment':1.8,'usefulness':2.2};config['class_weights']=None;config['rationale_sha256']={s:hashlib.sha256((root/'data'/f'{s}.json').read_bytes()).hexdigest() for s in data}
 if args.model=='cafe':
  from .cafe_training import make_optimizers,train_batch as cafe_train_batch
  cafe_optimizers=make_optimizers(model,profile['lr'],profile['weight_decay'])
  config['optimizer']='two alternating Adam optimizers: similarity then detection'
  config['cafe_features']='frozen Chinese BERT/ResNet18 plus trainable 200-d projection'
 config['implementation_revision']='2026-11-15-bert-head-cafe-two-stage-warmup-stop'
 dump(out/'config.json',config);best=-1;stale=0;history=[];started=time.time()
 for epoch in range(args.epochs):
  model.train();total_loss=0
  for batch in tqdm(loaders['train'],desc=f'{args.model}/{args.task} epoch {epoch+1}/{args.epochs}',unit='batch',dynamic_ncols=True,leave=False):
   b={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
   if args.model=='cafe':
    loss,details=cafe_train_batch(model,b,criterion,cafe_optimizers)
    total_loss+=loss.item()*len(b['target']);continue
   opt.zero_grad(set_to_none=True);o=model(b);loss=criterion(o['logits'],b['target'])
   if 'event_logits' in o:loss=loss+nn.functional.cross_entropy(o['event_logits'],b['event'])
   if 'aux_loss' in o:loss=loss+(.1 if args.model=='hammer' else 1.0)*o['aux_loss']
   if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
   loss.backward();opt.step();scheduler.step();total_loss+=loss.item()*len(b['target'])
  val,*_=evaluate(model,loaders['val'],device,codes);history.append({'epoch':epoch+1,'loss':total_loss/len(datasets['train']),'val':val});dump(out/'history.json',history)
  print(f"{args.model}/{args.task} epoch={epoch+1} loss={history[-1]['loss']:.4f} val_macro_f1={val['macro_f1']:.4f}",flush=True)
  improved=val['macro_f1']>best
  if improved:
   best=val['macro_f1'];stale=0;torch.save({'model':model.state_dict(),'epoch':epoch+1,'val':val,'config':config},out/'best.pt')
  stale=next_stale(stale,improved,epoch+1,profile.get('warmup_epochs',0))
  if args.patience and stale>=args.patience:break
 checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model']);test,ids,y,prob=evaluate(model,loaders['test'],device,codes);dump(out/'test_metrics.json',test)
 with (out/'test_predictions.jsonl').open('w',encoding='utf-8') as f:
  for i,actual,p in zip(ids,y,prob):f.write(json.dumps({'id':i,'label':codes[actual],'prediction':codes[int(np.argmax(p))],'probabilities':p},ensure_ascii=False)+'\n')
 import csv
 with (out/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
  columns=['id','label','prediction','correct']+[f'probability_{c}' for c in codes]
  if len(codes)>2:columns+=['binary_label','binary_prediction']
  writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
  for i,actual,p in zip(ids,y,prob):
   predicted=int(np.argmax(p));row={'id':i,'label':codes[actual],'prediction':codes[predicted],'correct':actual==predicted,**{f'probability_{c}':float(v) for c,v in zip(codes,p)}}
   if len(codes)>2:row.update(binary_label=int(codes[actual]!=0),binary_prediction=int(codes[predicted]!=0))
   writer.writerow(row)
 if args.task=='text_binary':test['binary_field_semantics'].update(real='text unmodified (label=0)',fake='text modified (label=1)')
 elif args.task=='mm_binary':test['binary_field_semantics'].update(real='no constructed forgery (mlabel=0)',fake='constructed forgery (mlabel=1)')
 dump(out/'test_metrics.json',test)
 print('Test metrics:',json.dumps(test,ensure_ascii=False),flush=True)
 config['epochs_actual']=len(history);config['seconds']=time.time()-started;dump(out/'config.json',config)
 dump(out/'status.json',{'status':'complete','best_epoch':checkpoint['epoch'],'test_macro_f1':test['macro_f1']});print('Completed',out,flush=True)
if __name__=='__main__':main()
