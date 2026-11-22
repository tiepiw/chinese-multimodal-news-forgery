"""Atomic epoch-boundary checkpoints; restores training and sampling state."""
import os,random
from pathlib import Path
import numpy as np
import torch

def atomic_save(payload,path):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 torch.save(payload,tmp);os.replace(tmp,path)

def save_training(path,model,optimizer,scheduler,scaler,generator,**progress):
 payload=dict(progress,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict() if scaler else None,rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,generator=generator.get_state())
 atomic_save(payload,path)

def restore_training(path,model,optimizer,scheduler,scaler,generator,contract):
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['contract']!=contract:raise ValueError('Resume data/settings differ. Use a separate --output-dir; checkpoint preserved.')
 model.load_state_dict(p['model']);optimizer.load_state_dict(p['optimizer']);scheduler.load_state_dict(p['scheduler'])
 if scaler and p['scaler'] is not None:scaler.load_state_dict(p['scaler'])
 random.setstate(p['rng_python']);np.random.set_state(p['rng_numpy']);torch.set_rng_state(p['rng_torch'])
 if torch.cuda.is_available() and p['rng_cuda'] is not None:torch.cuda.set_rng_state_all(p['rng_cuda'])
 generator.set_state(p['generator']);return p
