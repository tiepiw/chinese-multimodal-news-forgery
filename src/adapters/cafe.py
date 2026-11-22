"""CAFE-BERT feature adapter retaining the author's ambiguity/fusion modules."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence
from transformers import AutoModel
from .cafe_vendor import SimilarityModule,DetectionModule
from ..models import load_vision
class CAFE(nn.Module):
 def __init__(self,nclasses,bert_path,vision_path,random_init=False,bert=None,vision=None):
  super().__init__();self.bert=bert if bert is not None else AutoModel.from_pretrained(str(bert_path),local_files_only=True);self.bert.requires_grad_(False)
  self.proj=nn.Linear(self.bert.config.hidden_size,200);self.vision=vision if vision is not None else load_vision('resnet18',vision_path,random_init);self.vision.fc=nn.Identity();self.vision.requires_grad_(False)
  self.sim=SimilarityModule();self.det=DetectionModule();self.det.classifier_corre[-1]=nn.Linear(64,nclasses)
 def train(self,mode=True):
  super().train(mode);self.bert.eval();self.vision.eval();return self
 def features(self,b):
  with torch.no_grad():tokens=self.bert(input_ids=b['input_ids'],attention_mask=b['attention_mask']).last_hidden_state
  text=self.proj(tokens)*b['attention_mask'].unsqueeze(-1);image=torch.zeros(len(text),512,device=text.device);valid=b['has_image'];has=valid.float().unsqueeze(1)
  if valid.any():
   with torch.no_grad():image[valid]=self.vision(b['image'][valid])
  return text,image
 def alignment_loss(self,b):
  text,image=self.features(b)
  clean=(b['forgery']==0)&b['has_image']
  idx=clean.nonzero().flatten()
  if len(idx)<2:return None
  text=text[idx].detach();image=image[idx].detach()
  shift=3 if len(idx)>3 else 1
  t1,i1,_=self.sim(text,image);t2,i2,_=self.sim(text,image.roll(shift,0))
  return F.cosine_embedding_loss(torch.cat([t1,t2]),torch.cat([i1,i2]),torch.cat([torch.ones(len(idx),device=text.device),-torch.ones(len(idx),device=text.device)]))
 def forward(self,b):
  text,image=self.features(b);has=b['has_image'].float().unsqueeze(1)
  ta,ia,_=self.sim(text,image)
  # Use the author's sampled ambiguity calculation in both train and eval.
  ambiguity=self.det.ambiguity_module(ta,ia)
  ts,ims=self.det.encoding(text,image);tu,iu=self.det.uni_repre(ts,ims)
  correlation=self.det.cross_module(ta,ia)
  w=ambiguity.unsqueeze(1)*has
  logits=self.det.classifier_corre(torch.cat([(1-w)*tu,(1-w)*iu*has,w*correlation],1))
  return {'logits':logits}
