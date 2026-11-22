"""Verify generated ARG fields against raw time-split records, with no row dropping."""
import json
from pathlib import Path

def load_arg_rows(path,raw):
 path=Path(path)
 if not path.is_file():raise FileNotFoundError('Packaged ARG auxiliary fields not found: '+str(path))
 rows=json.loads(path.read_text(encoding='utf-8'))
 if len(rows)!=len(raw):raise ValueError('Incomplete ARG split: '+str(path))
 for r,source in zip(rows,raw):
  if any(r.get(k)!=source.get(k) for k in ['id','content','label']):raise ValueError('ARG row does not match the current time split: '+str(r.get('id')))
  for view in ['td','cs']:
   if not isinstance(r.get(view+'_rationale'),str) or not r[view+'_rationale'].strip():raise ValueError('Missing rationale: '+str(r['id']))
   pred=r.get(view+'_pred');acc=r.get(view+'_acc')
   if type(pred) is not int or pred not in [0,1] or type(acc) is not int or acc!=int(pred==r['label']):raise ValueError('Invalid ARG auxiliary label: '+str(r['id']))
 return rows
