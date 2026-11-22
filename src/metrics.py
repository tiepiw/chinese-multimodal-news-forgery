import numpy as np
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report


def binary_fields(y,pred):
 report=classification_report(y,pred,labels=[0,1],output_dict=True,zero_division=0)
 real,fake,macro=report['0'],report['1'],report['macro avg']
 return {'metric':float(macro['f1-score']),'f1_real':float(real['f1-score']),'f1_fake':float(fake['f1-score']),
         'recall':float(macro['recall']),'recall_real':float(real['recall']),'recall_fake':float(fake['recall']),
         'precision':float(macro['precision']),'precision_real':float(real['precision']),'precision_fake':float(fake['precision']),
         'acc':float(accuracy_score(y,pred))}


def metrics(y,prob,codes):
 pred=np.asarray(prob).argmax(1);y=np.asarray(y);indices=list(range(len(codes)))
 result={'samples':len(y),'macro_f1':float(f1_score(y,pred,labels=indices,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,pred)),'label_codes':codes,'confusion_matrix':confusion_matrix(y,pred,labels=indices).tolist(),'classification_report':classification_report(y,pred,labels=indices,target_names=[str(c) for c in codes],output_dict=True,zero_division=0)}
 if len(codes)>2:
  binary_y=(y!=codes.index(0)).astype(int);binary_pred=(pred!=codes.index(0)).astype(int)
  fields=binary_fields(binary_y,binary_pred)
  result['collapsed_binary']={**fields,'macro_f1':fields['metric'],'accuracy':fields['acc'],'confusion_matrix':confusion_matrix(binary_y,binary_pred,labels=[0,1]).tolist(),'label_semantics':{'real':'code 0','fake':'all nonzero forgery codes'},'prediction_rule':'collapse multiclass argmax; not probability thresholding'}
 else:
  result.update(binary_fields((y!=codes.index(0)).astype(int),(pred!=codes.index(0)).astype(int)))
  result['binary_field_semantics']={'real':'target code 0','fake':'target code 1','metric':'macro F1','recall':'macro recall','precision':'macro precision'}
 return result
