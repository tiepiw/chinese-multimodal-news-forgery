"""Early stopping counts completed epochs only after learning-rate warmup."""
def next_stale(stale,improved,completed_epoch,warmup_epochs):
 if improved or completed_epoch<=warmup_epochs:return 0
 return stale+1
