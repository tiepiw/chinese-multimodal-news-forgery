"""CAFE similarity step followed by detection step, with disjoint optimizers."""
import torch

def make_optimizers(model,lr,weight_decay):
 similarity=torch.optim.Adam(model.sim.parameters(),lr=lr,weight_decay=weight_decay)
 detection=torch.optim.Adam(list(model.det.parameters())+list(model.proj.parameters()),lr=lr,weight_decay=weight_decay)
 return similarity,detection

def train_batch(model,batch,criterion,optimizers):
 similarity,detection=optimizers
 # No labels enter inference. Clean training pairs supervise alignment only.
 similarity.zero_grad(set_to_none=True);detection.zero_grad(set_to_none=True)
 alignment=model.alignment_loss(batch)
 if alignment is not None:
  if not torch.isfinite(alignment):raise RuntimeError('Non-finite CAFE alignment loss')
  alignment.backward();similarity.step()
 # Recompute after the alignment update, as in the original training loop.
 similarity.zero_grad(set_to_none=True);detection.zero_grad(set_to_none=True)
 output=model(batch);loss=criterion(output['logits'],batch['target'])
 if not torch.isfinite(loss):raise RuntimeError('Non-finite CAFE detection loss')
 loss.backward();detection.step()
 return loss.detach(),{'alignment_updated':alignment is not None,'alignment_loss':float(alignment.detach()) if alignment is not None else None}
