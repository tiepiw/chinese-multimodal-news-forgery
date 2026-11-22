"""Source-backed training defaults; adapter choices are explicitly identified."""
from copy import deepcopy
PROFILES = {
 'arg': dict(epochs=50,batch_size=64,max_length=300,patience=5,lr=2e-5,weight_decay=5e-5,optimizer='Adam',schedule='constant',source='ARG implementation: Chinese configuration and Adam optimizer'),
 'bert': dict(epochs=3,batch_size=32,max_length=128,patience=0,lr=5e-5,weight_decay=.01,optimizer='AdamW',schedule='linear',source='google-research/bert: run_classifier.py and optimization.py'),
 'textcnn': dict(epochs=25,batch_size=50,max_length=None,patience=0,lr=1.,weight_decay=0.,optimizer='Adadelta',schedule='constant',source='yoonkim/CNN_sentence: conv_net_sentence.py',note='Character tokenizer adaptation; length derived from training data, not an author fixed length. Random embeddings replace unavailable source word vectors.'),
 'eann': dict(epochs=100,batch_size=100,max_length=None,patience=0,lr=.001,weight_decay=0.,optimizer='Adam',schedule='constant',source='yaqingwang/EANN-KDD18: src/EANN.py',note='Character embeddings and train-only pseudo events are adaptations; source length comes from its word-embedding pickle.'),
 'cafe': dict(epochs=100,batch_size=64,max_length=128,patience=0,lr=.001,weight_decay=0.,optimizer='Adam',schedule='constant',source='CAFE: train.py',note='128 is an adapter choice for BERT feature extraction; author code reads precomputed features and supplies no tokenizer limit. Two alternating Adam optimizers train alignment and detection.'),
 'hammer': dict(epochs=50,batch_size=32,max_length=128,patience=0,lr=2e-5,vision_lr=1e-4,weight_decay=.02,optimizer='AdamW',schedule='cosine',source='HAMMER: configs/train.yaml and train.py',note='Classification adaptation; cosine schedule is stepped per batch, warmup 10 epochs, minimum lr 1e-6.'),
}
ADAPTER_DEFAULT = dict(epochs=12,batch_size=64,max_length=128,patience=4,lr=1e-4,bert_lr=2e-5,vision_lr=1e-4,weight_decay=.01,optimizer='AdamW',schedule='linear',source='Project-defined reference baseline, not author defaults')
def resolve_profile(model,overrides=None,train_rows=None):
 p=deepcopy(PROFILES.get(model,ADAPTER_DEFAULT))
 p['source_batch_size']=p['batch_size'] if model in PROFILES else None
 if model in ['arg','bert','textcnn']:p['batch_size']=64
 elif model in ['concat','eann','cafe','hammer']:p['batch_size']=32
 p['batch_size_policy']='Benchmark protocol: text models 64, visual/multimodal models 32, otherwise 64; explicit CLI overrides take precedence.'
 p['warmup_epochs']=10 if model=='hammer' else 0
 p['source_patience']=p['patience']
 p['patience']=5
 p['early_stopping_policy']='Validation macro-F1; count patience only after warmup; stop after 5 consecutive epochs without strict improvement; test best validation checkpoint only.'
 for key in ('epochs','batch_size','max_length','patience','lr','bert_lr','vision_lr','weight_decay','warmup_epochs','optimizer','schedule'):
  value=(overrides or {}).get(key)
  if value is not None:p[key]=value
 if p['max_length'] is None and train_rows is not None:
  p['max_length']=max(5,max(len(row['content']) for row in train_rows))
 return p
