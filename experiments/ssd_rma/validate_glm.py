"""Validate GLM dimensions and post-projection math with synthetic inputs."""
import json
import os
import tempfile
import numpy as np
import onnxruntime as ort
import torch
from build import ROOT
from build_glm import build_glm

def session(name):
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=2
    return ort.InferenceSession(str(ROOT/(name+'.onnx')),opts,providers=['CPUExecutionProvider'])

def run():
    build_glm()
    sel,read=session('glm53_dsa_select'),session('glm53_mla_read')
    torch.set_num_threads(2)
    rng=np.random.default_rng(53)
    error=0.; cases=0; max_read=0.
    for b,s,n,k,mode in [(2,2,33,8,'normal'),(1,1,2053,2048,'normal'),
                          (1,1,17,2048,'ties'),(1,1,7,4,'empty')]:
        q=rng.normal(size=(b,s,32,128)).astype('f')
        keys=rng.normal(size=(b,n,128)).astype('f')
        hw=rng.normal(size=(b,s,32)).astype('f')
        allowed=np.ones((b,s,n),dtype=bool)
        if mode=='normal' and n<100:
            allowed[:,0,n//2:]=False  # simulate earlier query's causal bound
        if mode=='empty': allowed.fill(False)
        if mode=='ties': hw.fill(0)
        stride=4096  # FP32 prototype record payload 576*4 bytes
        base=np.zeros((b,1,1),dtype='int64')
        feed=dict(index_query=q,index_key=keys,head_weights=hw,
            allowed=allowed,requested_k=np.array([k],dtype='int64'),
            record_stride=np.array(stride,dtype='int64'),base_offsets=base)
        top,slots,valid,offsets=sel.run(None,feed)
        if b==1 and n==2053:
            np.savez(ROOT/'glm53_dsa_inputs.npz',**feed)
            np.savez(ROOT/'glm53_dsa_expected.npz',top_scores=top,slots=slots,
                     selected_valid=valid,byte_offsets=offsets)
        ref=torch.relu(torch.from_numpy(q) @ torch.from_numpy(keys).transpose(-1,-2)[:,None] * 128**-.5)
        ref=(torch.from_numpy(hw)*32**-.5).unsqueeze(-2).matmul(ref).squeeze(-2)
        ref=ref.masked_fill(~torch.from_numpy(allowed),-torch.inf)
        tv,ti=ref.topk(min(k,n,2048),dim=-1)
        np.testing.assert_allclose(top,tv.numpy(),atol=4e-5,rtol=1e-4)
        finite=np.isfinite(top)
        assert np.array_equal(valid,finite)
        assert np.all(slots[~valid]==-1) and np.all(offsets[~valid]==-1)
        np.testing.assert_array_equal(offsets[valid],slots[valid]*stride)
        if mode=='normal':
            np.testing.assert_array_equal(slots[valid],ti.numpy()[valid])
        if mode=='ties':
            # ONNX prefers smaller input indices; PyTorch topk ties unspecified.
            np.testing.assert_array_equal(slots,np.arange(n,dtype='int64')[None,None])
        error=max(error,float(np.max(np.abs(top[finite]-tv.numpy()[finite]))) if finite.any() else 0.)
        k_eff=slots.shape[-1]
        storage=rng.normal(size=(b,n,576)).astype('f')
        selected=np.zeros((b,s,k_eff,576),dtype='f')
        with tempfile.TemporaryFile(dir=ROOT) as f:
            for bi in range(b):
                for ni in range(n):
                    payload=storage[bi,ni].tobytes()
                    f.write(payload+bytes(stride-len(payload)))
            f.flush(); os.fsync(f.fileno())
            for bi,si,ki in np.argwhere(valid):
                # base_offsets is per batch sequence file; same file here has b regions.
                pos=bi*n*stride+int(offsets[bi,si,ki])
                selected[bi,si,ki]=np.frombuffer(os.pread(f.fileno(),576*4,pos),dtype='f')
        for bi,si,ki in np.argwhere(valid):
            np.testing.assert_array_equal(selected[bi,si,ki],storage[bi,slots[bi,si,ki]])
        qa=(rng.normal(size=(b,s,64,512))*.1).astype('f')
        qr=(rng.normal(size=(b,s,64,64))*.1).astype('f')
        lat=selected[...,:512].copy(); rk=selected[...,512:].copy()
        out=read.run(None,dict(latent=lat,rope_key=rk,query_absorbed=qa,query_rope=qr,selected_valid=valid))[0]
        logits=(torch.from_numpy(qa) @ torch.from_numpy(lat).transpose(-1,-2)
            +torch.from_numpy(qr) @ torch.from_numpy(rk).transpose(-1,-2))*256**-.5
        mask=torch.from_numpy(valid)[:,:,None,:]
        logits=logits.masked_fill(~mask,-torch.inf)
        logits=torch.where(mask.any(-1,keepdim=True),logits,0.)
        weights=torch.where(mask,logits.softmax(-1),0.)
        want=(weights @ torch.from_numpy(lat)).numpy()
        np.testing.assert_allclose(out,want,atol=4e-6,rtol=1e-4)
        max_read=max(max_read,float(np.max(np.abs(out-want))))
        if mode=='empty': assert not out.any()
        cases+=1
    report=dict(status='PASS',cases=cases,index_heads=32,index_dim=128,mla_rank=512,
        rope_dim=64,attention_heads=64,tested_topk_2048=True,max_score_error=error,
        max_latent_read_error=max_read,
        scope='FP32 synthetic post-projection DSA and absorbed MLA, CPU file I/O',
        not_validated=['checkpoint weights and BF16/FP8 numerics','GLM end-to-end generation',
                       'ARM execution','GDS direct path','SSD physical device','performance'])
    (ROOT/'glm_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': run()
