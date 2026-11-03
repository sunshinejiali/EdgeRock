"""Algorithm and file-interface validation, CPU only; NOT a GDS/SSD benchmark."""
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import onnxruntime as ort
import torch
from build import build, ROOT

sys.path.insert(0, str(ROOT.parents[1]))
from modeling_edgerock import select_topk

def session(name):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(str(ROOT / (name + '.onnx')), opts,
                               providers=['CPUExecutionProvider'])

def run():
    build()
    rng = np.random.default_rng(42)
    scorer, selector, reader = [session(n) for n in
                               ('admission_score','stable_select','memory_read')]
    cases = 0
    max_score_error = max_read_error = 0.0
    torch.set_num_threads(2)
    for b,g,n,h,r,d,k in [(1,2,17,32,8,16,5),(2,3,31,48,12,8,7),(1,1,8,16,4,8,8)]:
        u = rng.normal(size=(b,n,h)).astype('f')
        down = (rng.normal(size=(h,r)) * .1).astype('f')
        up = (rng.normal(size=(r,g)) * .1).astype('f')
        scores = scorer.run(None,dict(u=u,down=down,up=up))[0]
        expected = (torch.nn.functional.silu(torch.from_numpy(u) @ torch.from_numpy(down))
                    @ torch.from_numpy(up)).permute(0,2,1).numpy()
        np.testing.assert_allclose(scores,expected,atol=2e-6,rtol=2e-5)
        max_score_error=max(max_score_error,float(np.max(np.abs(scores-expected))))
        positions=np.broadcast_to(rng.permutation(n).astype('int64'),(b,g,n)).copy()
        refs=np.broadcast_to(np.arange(n,dtype='int64'),(b,g,n)).copy()
        kv=rng.normal(size=(b,g,n,2,d)).astype('f')
        query=rng.normal(size=(b,g,2,d)).astype('f')
        # unique scores, equal scores with shuffled positions, partially/all invalid
        for mode in ('ordinary','ties','partial_invalid','all_invalid'):
            s=scores.copy(); ref=refs.copy()
            if mode=='ties': s.fill(0)
            if mode=='partial_invalid': ref[...,::2]=-1
            if mode=='all_invalid': ref.fill(-1)
            s[ref<0]=-np.inf
            feed=dict(scores=s,positions=positions,references=ref,capacity=np.array([k],dtype='int64'))
            selected=selector.run(None,feed)
            expected=select_topk(*[torch.from_numpy(v) for v in (s,positions,ref)],k)
            for actual,want in zip(selected[:3],expected):
                np.testing.assert_array_equal(actual,want.numpy())
            valid=selected[2]>=0
            chosen=np.zeros((b,g,k,2,d),dtype='f')
            # Row layout [batch, KV group, token, K/V, dim], one 4 KiB record.
            # Ordinary buffered file I/O through CPU. Hardware medium unknown.
            stride=4096
            assert 2*d*4<=stride
            with tempfile.TemporaryFile(dir=ROOT) as f:
                for bi in range(b):
                    for gi in range(g):
                        for ti in range(n):
                            raw=kv[bi,gi,ti].tobytes()
                            f.write(raw+bytes(stride-len(raw)))
                f.flush(); os.fsync(f.fileno())
                for bi,gi,ki in np.argwhere(valid):
                    ti=int(selected[2][bi,gi,ki])
                    offset=((bi*g+gi)*n+ti)*stride
                    raw=os.pread(f.fileno(),2*d*4,offset)
                    chosen[bi,gi,ki]=np.frombuffer(raw,dtype='f').reshape(2,d)
            want_kv=np.zeros_like(chosen)
            for bi,gi,ki in np.argwhere(valid):
                want_kv[bi,gi,ki]=kv[bi,gi,selected[2][bi,gi,ki]]
            np.testing.assert_array_equal(chosen,want_kv)
            out=reader.run(None,dict(query=query,keys=chosen[:,:,:,0,:].copy(),
                                    values=chosen[:,:,:,1,:].copy(),valid=valid,
                                    scale=np.array(d**-.5,dtype='f')))[0]
            q=torch.from_numpy(query)
            logits=q @ torch.from_numpy(want_kv[:,:,:,0,:]).transpose(-1,-2) * d**-.5
            logits=logits.masked_fill(~torch.from_numpy(valid)[:,:,None,:],-torch.inf)
            weights=torch.nn.functional.pad(logits,(0,1),value=0).softmax(-1)[...,:-1]
            want=(weights @ torch.from_numpy(want_kv[:,:,:,1,:])).numpy()
            np.testing.assert_allclose(out,want,atol=2e-6,rtol=2e-5)
            assert np.isfinite(out).all()
            if mode=='all_invalid': np.testing.assert_array_equal(out,np.zeros_like(out))
            max_read_error=max(max_read_error,float(np.max(np.abs(out-want))))
            cases+=1
    # Two commits: merge retained metadata with new candidate metadata.
    s=rng.normal(size=(1,2,16)).astype('f')
    p=np.broadcast_to(np.arange(16,dtype='int64'),s.shape).copy()
    first=selector.run(None,dict(scores=s[...,:8],positions=p[...,:8],references=p[...,:8],capacity=np.array([4],dtype='int64')))
    merged=selector.run(None,dict(scores=np.concatenate([first[0],s[...,8:]],axis=-1),
        positions=np.concatenate([first[1],p[...,8:]],axis=-1),
        references=np.concatenate([first[2],p[...,8:]],axis=-1),capacity=np.array([4],dtype='int64')))
    full=select_topk(torch.from_numpy(s),torch.from_numpy(p),torch.from_numpy(p),4)
    for actual,want in zip(merged[:3],full): np.testing.assert_array_equal(actual,want.numpy())
    report=dict(status='PASS',shape_and_mask_cases=cases,incremental_merge='PASS',
        max_score_absolute_error=max_score_error,max_read_absolute_error=max_read_error,
        onnxruntime=ort.__version__,providers=['CPUExecutionProvider'],
        io='CPU buffered pread/pwrite equivalent; file writes plus pread; NOT GDS',
        weights='synthetic, not GLM or trained RMA weights',
        not_validated=['GLM 5.3 integration/quality','computational SSD execution',
                       'GPU-direct I/O','physical SSD medium','latency or throughput improvement'])
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': run()
