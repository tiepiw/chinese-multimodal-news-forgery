"""HAMMER classification adaptation using original ViT and fusion BERT.
No bbox/token localization loss: target dataset has no localization annotation.
"""
import copy
from pathlib import Path
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from .hammer_vendor.vit import VisionTransformer,interpolate_pos_embed
from .hammer_vendor.xbert import BertConfig,BertModel
class HAMMERClassifier(nn.Module):
 def __init__(self,nclasses,bert_path,vision_path,random_init=False,tiny_config=None):
  super().__init__()
  cfg=BertConfig(**tiny_config) if tiny_config else BertConfig.from_pretrained(str(bert_path),local_files_only=True)
  width=cfg.hidden_size;cfg.fusion_layer=getattr(cfg,'fusion_layer',cfg.num_hidden_layers//2);cfg.encoder_width=width
  self.text=BertModel(cfg) if random_init else BertModel.from_pretrained(str(bert_path),config=cfg,local_files_only=True,add_pooling_layer=False)
  size=32 if tiny_config else 256;depth=2 if tiny_config else 12
  self.vision=VisionTransformer(img_size=size,patch_size=16,embed_dim=width,depth=depth,num_heads=cfg.num_attention_heads,mlp_ratio=4,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6))
  if not random_init:
   path=Path(vision_path)
   if not path.is_file():raise FileNotFoundError('Place DeiT base weights at '+str(path))
   state=torch.load(path,map_location='cpu',weights_only=True);state=state.get('model',state);state={k:v for k,v in state.items() if not k.startswith(('head.','head_dist.'))};state['pos_embed']=interpolate_pos_embed(state['pos_embed'],self.vision)
   missing,unexpected=self.vision.load_state_dict(state,strict=False)
   if missing or unexpected:raise ValueError(f'DeiT checkpoint mismatch: {missing}, {unexpected}')
  dim=64 if tiny_config else 256
  self.vision_proj=nn.Linear(width,dim);self.text_proj=nn.Linear(width,dim)
  self.head=nn.Sequential(nn.Linear(width,width*2),nn.LayerNorm(width*2),nn.GELU(),nn.Linear(width*2,width*2),nn.LayerNorm(width*2),nn.GELU(),nn.Linear(width*2,nclasses))
  self.temp=nn.Parameter(torch.tensor(.07));self.momentum=.995
  self.vision_m=copy.deepcopy(self.vision).requires_grad_(False);self.text_m=copy.deepcopy(self.text).requires_grad_(False);self.vp_m=copy.deepcopy(self.vision_proj).requires_grad_(False);self.tp_m=copy.deepcopy(self.text_proj).requires_grad_(False)
  self.register_buffer('image_queue',F.normalize(torch.randn(dim,1024 if tiny_config else 65536),dim=0));self.register_buffer('text_queue',F.normalize(torch.randn(dim,1024 if tiny_config else 65536),dim=0));self.register_buffer('queue_ptr',torch.zeros(1,dtype=torch.long))
 def train(self,mode=True):
  super().train(mode)
  for m in [self.vision_m,self.text_m,self.vp_m,self.tp_m]:m.eval()
  return self
 @torch.no_grad()
 def update(self):
  for a,b in [(self.vision,self.vision_m),(self.text,self.text_m),(self.vision_proj,self.vp_m),(self.text_proj,self.tp_m)]:
   for p,q in zip(a.parameters(),b.parameters()):q.mul_(self.momentum).add_(p,alpha=1-self.momentum)
 def forward(self,b):
  valid=b['has_image'];iv=self.vision(b['image']);iv=iv*valid[:,None,None]
  imask=torch.ones(iv.shape[:2],dtype=torch.long,device=iv.device);imask[~valid]=0;imask[~valid,0]=1
  tx=self.text(b['input_ids'],attention_mask=b['attention_mask'],return_dict=True,mode='text').last_hidden_state
  fused=self.text(encoder_embeds=tx,attention_mask=b['attention_mask'],encoder_hidden_states=iv,encoder_attention_mask=imask,return_dict=True,mode='fusion').last_hidden_state
  logits=self.head(fused[:,0]);aux=logits.sum()*0
  if self.training and valid.any():
   with torch.no_grad():self.temp.clamp_(.001,.5);self.update()
   image_feat=F.normalize(self.vision_proj(iv[valid,0]),dim=-1);text_feat=F.normalize(self.text_proj(tx[valid,0]),dim=-1)
   with torch.no_grad():
    mi=F.normalize(self.vp_m(self.vision_m(b['image'][valid])[:,0]),dim=-1)
    mt=F.normalize(self.tp_m(self.text_m(b['input_ids'][valid],attention_mask=b['attention_mask'][valid],return_dict=True,mode='text').last_hidden_state[:,0]),dim=-1)
    ai=torch.cat([mi.T,self.image_queue.clone().detach()],1);at=torch.cat([mt.T,self.text_queue.clone().detach()],1)
    target=torch.zeros(len(mi),at.shape[1],device=mi.device);idx=torch.arange(len(mi),device=mi.device);target[idx,idx]=(b['forgery'][valid]==0).float()
    within=torch.zeros_like(target);within[idx,idx]=1
    # Original momentum distillation blend, fixed alpha=.4.
    it=.4*(mi@at/self.temp).softmax(-1)+.6*target;ti=.4*(mt@ai/self.temp).softmax(-1)+.6*target
   loss=lambda x,y:-(x.log_softmax(-1)*y).sum(-1).mean()
   aux=(loss(image_feat@at/self.temp,it)+loss(text_feat@ai/self.temp,ti)+loss(image_feat@ai/self.temp,within)+loss(text_feat@at/self.temp,within))/4
   with torch.no_grad():
    for j in range(len(mi)):
     ptr=int(self.queue_ptr.item());self.image_queue[:,ptr]=mi[j];self.text_queue[:,ptr]=mt[j];self.queue_ptr[0]=(ptr+1)%self.image_queue.shape[1]
  return {'logits':logits,'aux_loss':aux}
