"""Minimal CPU/ARM runner; no PyTorch or EdgeRock import required."""
import argparse
import time
import numpy as np
import onnxruntime as ort

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--inputs',required=True)
p.add_argument('--outputs',required=True)
p.add_argument('--threads',type=int,default=2)
a=p.parse_args()
opts=ort.SessionOptions()
opts.intra_op_num_threads=a.threads
s=ort.InferenceSession(a.model,opts,providers=['CPUExecutionProvider'])
with np.load(a.inputs,allow_pickle=False) as f:
    feed={i.name:f[i.name] for i in s.get_inputs()}
if 'requested_k' in feed:
    assert feed['requested_k'].shape==(1,) and feed['requested_k'][0]>=1
    assert feed['index_key'].shape[1]>=1
    assert feed['record_stride'].item()>0
    assert np.all(feed['base_offsets']>=0)
start=time.perf_counter()
result=s.run(None,feed)
elapsed=time.perf_counter()-start
np.savez(a.outputs,**{o.name:v for o,v in zip(s.get_outputs(),result)})
print(f'Executed with CPUExecutionProvider: {elapsed:.6f}s (one call, not a benchmark)')
