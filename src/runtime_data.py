import json
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset
from PIL import Image,ImageFile
from torchvision import transforms
from .data import image_references,resolve_image
from .registry import TASKS

def build_vocab(rows,limit=15000):
 counts=Counter(c for r in rows for c in r['content'])
 return {'<pad>':0,'<unk>':1,**{c:i+2 for i,(c,_) in enumerate(sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:limit-2])}}
class NewsDataset(Dataset):
 def __init__(self,rows,task,root,tokenizer=None,vocab=None,max_length=256,image_size=224,use_image=False,events=None):
  self.rows=rows;self.root=Path(root);self.tokenizer=tokenizer;self.vocab=vocab;self.length=max_length;self.use_image=use_image;self.events=events
  self.field,self.codes=TASKS[task];self.mapping={c:i for i,c in enumerate(self.codes)}
  self.transform=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))]);self.size=image_size
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];text=r['content']
  if self.tokenizer:
   enc=self.tokenizer(text,max_length=self.length,padding='max_length',truncation=True,return_tensors='pt');ids=enc['input_ids'][0];mask=enc['attention_mask'][0]
  else:
   values=[self.vocab.get(c,1) for c in text[:self.length]];ids=torch.tensor(values+[0]*(self.length-len(values)));mask=(ids!=0).long()
  image=torch.zeros(3,self.size,self.size);has=False
  if self.use_image:
   refs=image_references(r)
   for ref in refs:
    path=resolve_image(self.root,ref)
    if not path.is_file():raise FileNotFoundError('Missing referenced image: '+str(path))
    try:
     old_tolerance=ImageFile.LOAD_TRUNCATED_IMAGES
     try:
      if ref=='7557/tampered.jpg':ImageFile.LOAD_TRUNCATED_IMAGES=True
      with Image.open(path) as im:image=self.transform(im.convert('RGB'));has=True
     finally:ImageFile.LOAD_TRUNCATED_IMAGES=old_tolerance
     break
    except (OSError,ValueError) as e:raise RuntimeError('Unreadable referenced image: '+str(path)) from e
  return {'input_ids':ids,'attention_mask':mask,'image':image,'has_image':torch.tensor(has),'target':torch.tensor(self.mapping[int(r[self.field])]),'forgery':torch.tensor(int(r['forgery'])),'event':torch.tensor(int(self.events[i]) if self.events is not None else 0),'id':str(r['id'])}

def collate(batch):
 # Stock collation preserves strings as a list and stacks tensor fields.
 from torch.utils.data._utils.collate import default_collate
 return default_collate(batch)

class ARGDataset(NewsDataset):
 def __getitem__(self,i):
  item=super().__getitem__(i);row=self.rows[i]
  for v in ['td','cs']:
   enc=self.tokenizer(row[v+'_rationale'],max_length=self.length,padding='max_length',truncation=True,return_tensors='pt')
   item[v+'_ids']=enc['input_ids'][0];item[v+'_mask']=enc['attention_mask'][0]
   item[v+'_pred']=torch.tensor(row[v+'_pred'],dtype=torch.long);item[v+'_acc']=torch.tensor(row[v+'_acc'],dtype=torch.long)
  return item
