TASKS={'text_binary':('label',[0,1]),'mm_binary':('mlabel',[0,1]),'forgery5':('forgery',[0,101,102,110,111])}
TEXT_MODELS=['textcnn','bert','arg']
MM_MODELS=['concat','eann','cafe','hammer']
DEFAULT_MODELS=['textcnn','bert','concat','eann','cafe','hammer','arg','himf_moe','qwen35_4b']
def experiments(models,tasks):
 jobs=[]
 for m in models:
  if m=='himf_moe':
   if any(t not in TASKS for t in tasks):raise ValueError('Unknown task')
   if ('himf_moe','joint') not in jobs:jobs.append(('himf_moe','joint'))
   continue
  if m=='qwen35_4b':
   for t in tasks:
    if t not in TASKS:raise ValueError('Unknown task: '+t)
    if (m,t) not in jobs:jobs.append((m,t))
   continue
  if m not in TEXT_MODELS+MM_MODELS:raise ValueError('Unknown model: '+m)
  for t in tasks:
   if t not in TASKS:raise ValueError('Unknown task: '+t)
   if t=='text_binary' and m not in TEXT_MODELS:continue
   if t!='text_binary' and m in ['textcnn','arg']:continue
   jobs.append((m,t))
 if not jobs:raise ValueError('Selected models do not support the selected tasks')
 return jobs
