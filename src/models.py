"""Small baselines and method adapters; see THIRD_PARTY_NOTICES.md."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from transformers import AutoModel

class GradReverse(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,scale):ctx.scale=scale;return x.view_as(x)
 @staticmethod
 def backward(ctx,g):return -ctx.scale*g,None

def load_vision(kind,checkpoint,random_init=False):
 model=models.resnet18(weights=None) if kind=='resnet18' else models.vgg19(weights=None)
 if not random_init:
  path=Path(checkpoint)
  if not path.is_file():raise FileNotFoundError('Place pretrained weights at '+str(path))
  state=torch.load(path,map_location='cpu',weights_only=True)
  model.load_state_dict(state)
 return model
class TextCNN(nn.Module):
 def __init__(self,vocab_size,nclasses,dim=300,channels=100,kernels=(3,4,5)):
  super().__init__();self.embed=nn.Embedding(vocab_size,dim,padding_idx=0);self.convs=nn.ModuleList([nn.Conv1d(dim,channels,k) for k in kernels]);self.drop=nn.Dropout(.5);self.fc=nn.Linear(channels*len(kernels),nclasses)
 def encode(self,ids,mask):
  x=self.embed(ids)*mask.unsqueeze(-1);x=x.transpose(1,2);features=[]
  for conv in self.convs:
   y=F.relu(conv(x));k=conv.kernel_size[0];valid=mask.unfold(1,k,1).any(-1);y=y.masked_fill(~valid.unsqueeze(1),-1e4);features.append(y.max(-1).values)
  return self.drop(torch.cat(features,1))
 def forward(self,b):return {'logits':self.fc(self.encode(b['input_ids'],b['attention_mask']))}
class Basic(nn.Module):
 def __init__(self,name,nclasses,bert_path,vision_path,random_init=False,bert=None,vision=None):
  super().__init__()
  if name not in ['bert','concat']:raise ValueError('Unsupported baseline: '+name)
  self.name=name
  self.text=bert if bert is not None else AutoModel.from_pretrained(str(bert_path),local_files_only=True)
  self.tp=nn.Sequential(nn.Linear(self.text.config.hidden_size,256),nn.ReLU(),nn.LayerNorm(256))
  if name=='concat':
   self.vision=vision if vision is not None else load_vision('resnet18',vision_path,random_init);self.vision.fc=nn.Identity();self.ip=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.LayerNorm(256))
  self.drop=nn.Dropout(.1 if name=='bert' else .2)
  if name=='bert':self.tp=nn.Identity()
  self.head=nn.Linear(512 if name=='concat' else self.text.config.hidden_size,nclasses)
 def forward(self,b):
  has=b['has_image'].float().unsqueeze(1)
  encoded=self.text(input_ids=b['input_ids'],attention_mask=b['attention_mask'])
  t=encoded.pooler_output if self.name=='bert' else self.tp(encoded.last_hidden_state[:,0])
  if self.name=='bert':out=self.head(self.drop(t))
  else:
   raw=torch.zeros(b['image'].shape[0],512,device=b['image'].device)
   valid=b['has_image']
   if valid.any():raw[valid]=self.vision(b['image'][valid])
   im=self.ip(raw)*has
   out=self.head(self.drop(torch.cat([t,im],1)))
  return {'logits':out}
class EANN(nn.Module):
 """Port of upstream CNN_Fusion: masked embedding/CNN + frozen VGG19 + GRL."""
 def __init__(self,vocab_size,nclasses,vision_path,event_count=20,random_init=False,vision=None):
  super().__init__();self.embed=nn.Embedding(vocab_size,256,padding_idx=0);self.convs=nn.ModuleList([nn.Conv1d(256,20,k) for k in [1,2,3,4]]);self.text_fc=nn.Linear(80,32)
  self.vgg=vision if vision is not None else load_vision('vgg19',vision_path,random_init)
  self.vgg.requires_grad_(False);self.vgg.eval();self.image_fc=nn.Linear(1000,32);self.head=nn.Linear(64,nclasses);self.event_head=nn.Sequential(nn.Linear(64,32),nn.LeakyReLU(),nn.Linear(32,event_count))
 def train(self,mode=True):super().train(mode);self.vgg.eval();return self
 def forward(self,b):
  x=(self.embed(b['input_ids'])*b['attention_mask'].unsqueeze(-1)).transpose(1,2)
  t=F.leaky_relu(self.text_fc(torch.cat([F.leaky_relu(c(x)).amax(-1) for c in self.convs],1)))
  raw=torch.zeros(len(t),1000,device=t.device);valid=b['has_image']
  if valid.any():
   with torch.no_grad():raw[valid]=self.vgg(b['image'][valid])
  im=F.leaky_relu(self.image_fc(raw))*valid.float().unsqueeze(1);f=torch.cat([t,im],1)
  return {'logits':self.head(f),'event_logits':self.event_head(GradReverse.apply(f,1.0))}
