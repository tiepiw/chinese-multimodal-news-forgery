"""ARG textual-modification adaptation; labels only supervise training losses."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig
from .arg_vendor.model import ARGModel
class ARGClassifier(nn.Module):
 def __init__(self,bert_path):
  super().__init__();dim=AutoConfig.from_pretrained(str(bert_path),local_files_only=True).hidden_size
  config={'bert_path':str(bert_path),'emb_dim':dim,'co_attention_dim':300,'model':{'mlp':{'dims':[384],'dropout':.2}}}
  self.core=ARGModel(config)
 def forward(self,b):
  # No ground-truth/predicted correctness labels enter the inference network.
  o=self.core(content=b['input_ids'],content_masks=b['attention_mask'],FTR_2=b['td_ids'],FTR_2_masks=b['td_mask'],FTR_3=b['cs_ids'],FTR_3_masks=b['cs_mask'])
  p=o['classify_pred'].clamp(1e-6,1-1e-6);logits=torch.stack([(1-p).log(),p.log()],1);aux=logits.sum()*0
  if self.training:
   hard=F.binary_cross_entropy(o['hard_ftr_2_pred'],b['td_acc'].float())+F.binary_cross_entropy(o['hard_ftr_3_pred'],b['cs_acc'].float())
   simple=F.cross_entropy(o['simple_ftr_2_pred'],b['td_pred'])+F.cross_entropy(o['simple_ftr_3_pred'],b['cs_pred'])
   aux=2.2*hard/2+1.8*simple/2
  return {'logits':logits,'aux_loss':aux}
