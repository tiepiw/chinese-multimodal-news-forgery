from pathlib import Path
from .models import Basic,TextCNN,EANN

def build_model(name,nclasses,root,vocab_size=15000,event_count=20):
 root=Path(root);p=root/'pretrained';bert=p/'chinese-bert-wwm-ext';res=p/'resnet18/resnet18-f37072fd.pth'
 if name=='arg':
  if nclasses!=2:raise ValueError('ARG is registered only for text_binary')
  from .adapters.arg import ARGClassifier
  return ARGClassifier(bert)
 if name=='textcnn':return TextCNN(vocab_size,nclasses)
 if name=='eann':return EANN(vocab_size,nclasses,p/'vgg19/vgg19-dcbb9e9d.pth',event_count)
 if name=='cafe':
  from .adapters.cafe import CAFE
  return CAFE(nclasses,bert,res)
 if name=='hammer':
  from .adapters.hammer import HAMMERClassifier
  return HAMMERClassifier(nclasses,bert,p/'deit/deit_base_patch16_224-b5f2ef4d.pth')
 return Basic(name,nclasses,bert,res)
