"""Pure-Python data contract; no training dependency required for audit."""
import re,json
from pathlib import Path
from datetime import datetime
from collections import Counter
CODES=[0,101,102,110,111]
def image_references(row):
 v=row.get('images')
 if isinstance(v,dict):
  def key(k):
   n=re.findall(r'\d+',str(k));return (int(n[-1]) if n else 0,str(k))
  return [str(v[k]) for k in sorted(v,key=key) if v[k]]
 if isinstance(v,list):return [str(x) for x in v if x]
 return [v] if isinstance(v,str) and v else []
def resolve_image(root,reference):
 root=Path(root).resolve();p=(root/reference).resolve()
 if p!=root and root not in p.parents:raise ValueError('Unsafe image path: '+reference)
 return p

def parse_date(row):
 m=re.search(r'(20\d{2})\D+(\d{1,2})\D+(\d{1,2})(?:\D+(\d{1,2}):(\d{2}))?',str(row.get('publish_time','')))
 if not m:raise ValueError('Missing date: '+str(row.get('id')))
 return datetime(*[int(x or 0) for x in m.groups()])
def load_splits(root):return {s:json.loads((Path(root)/(s+'.json')).read_text(encoding='utf-8')) for s in ['train','val','test']}
def audit_records(splits):
 seen=set();urls=set();bodies={};report={}
 for s in ['train','val','test']:
  rows=splits[s]
  if not rows:raise ValueError('Empty split: '+s)
  for r in rows:
   c=int(r['forgery']);i=str(r['id']);u=r.get('url')
   if c not in CODES or int(r['label'])!=int(c in [110,111]) or int(r['mlabel'])!=int(c!=0):raise ValueError('Invalid labels: '+i)
   if i in seen:raise ValueError('Repeated ID: '+i)
   if u and u in urls:raise ValueError('Repeated URL: '+str(u))
   seen.add(i);urls.add(u)
   text=re.sub(r'\s+','',r.get('content',''))
   if not text:raise ValueError('Empty body: '+i)
   if text in bodies and bodies[text]!=s:raise ValueError('Identical body across splits')
   bodies[text]=s
  # Dates are optional metadata in the six-field distribution.
  dated=[bool(r.get('publish_time')) for r in rows]
  if any(dated) and not all(dated):raise ValueError('Partially missing dates: '+s)
  dates=[parse_date(r) for r in rows] if all(dated) else []
  report[s]={'records':len(rows),'forgery':dict(Counter(int(r['forgery']) for r in rows)),'earliest':min(dates).isoformat() if dates else None,'latest':max(dates).isoformat() if dates else None}
 for a,b in [('train','val'),('val','test')]:
  if report[a]['latest'] is not None and report[b]['earliest'] is not None:
   if report[a]['latest']>=report[b]['earliest']:raise ValueError('Time overlap: '+a+'/'+b)
 return report
