"""GLM-5.3 config-specific DSA selection and absorbed MLA read contracts.

No checkpoint weights. Q/K projection, norms, RoPE and value/output projections
remain in the model. FP32 validation boundary; production BF16 conversion external.
"""
from build import save, vi, const, h, T

def build_glm():
    save('glm53_dsa_select', [
        h.make_node('Transpose',['index_key'],['kt'],perm=[0,2,1]),
        h.make_node('Unsqueeze',['kt','axis_s'],['kt4']),
        h.make_node('MatMul',['index_query','kt4'],['dot']),
        h.make_node('Mul',['dot','index_scale'],['scaled']),
        h.make_node('Relu',['scaled'],['positive']),
        h.make_node('Mul',['head_weights','head_scale'],['w']),
        h.make_node('Unsqueeze',['w','axis_h'],['w4']),
        h.make_node('MatMul',['w4','positive'],['score4']),
        h.make_node('Squeeze',['score4','axis_h'],['raw_scores']),
        h.make_node('Where',['allowed','raw_scores','neg_inf'],['scores']),
        h.make_node('Shape',['scores'],['shape']),
        h.make_node('Gather',['shape','last_axis'],['n'],axis=0),
        h.make_node('Min',['requested_k','max_k','n'],['effective_k']),
        h.make_node('TopK',['scores','effective_k'],['top_scores','raw_slots'],axis=-1,largest=1,sorted=1),
        h.make_node('GatherElements',['allowed','raw_slots'],['selected_valid'],axis=-1),
        h.make_node('Where',['selected_valid','raw_slots','minus_one'],['slots']),
        h.make_node('Mul',['raw_slots','record_stride'],['relative_offsets']),
        h.make_node('Add',['relative_offsets','base_offsets'],['raw_offsets']),
        h.make_node('Where',['selected_valid','raw_offsets','minus_one'],['byte_offsets']),
    ],[vi('index_query',T.FLOAT,['B','S',32,128]),
       vi('index_key',T.FLOAT,['B','N',128]),vi('head_weights',T.FLOAT,['B','S',32]),
       vi('allowed',T.BOOL,['B','S','N']),vi('requested_k',T.INT64,[1]),
       vi('record_stride',T.INT64,[]),vi('base_offsets',T.INT64,['B',1,1])],
       [vi('top_scores',T.FLOAT,['B','S','K']),vi('slots',T.INT64,['B','S','K']),
        vi('selected_valid',T.BOOL,['B','S','K']),vi('byte_offsets',T.INT64,['B','S','K'])],
       [const('axis_s',T.INT64,[1]),const('axis_h',T.INT64,[2]),
        const('index_scale',T.FLOAT,[128**-.5],[]),const('head_scale',T.FLOAT,[32**-.5],[]),
        const('last_axis',T.INT64,[2]),const('max_k',T.INT64,[2048]),
        const('neg_inf',T.FLOAT,[float('-inf')],[]),const('minus_one',T.INT64,[-1],[])])

    # q_abs[h] = q_nope[h] @ W_k[h]. Returned latent context must pass
    # through W_v[h] and o_proj on GPU. This graph is not full attention output.
    save('glm53_mla_read',[
        h.make_node('Transpose',['latent'],['ct'],perm=[0,1,3,2]),
        h.make_node('Transpose',['rope_key'],['rt'],perm=[0,1,3,2]),
        h.make_node('MatMul',['query_absorbed','ct'],['nope_logits']),
        h.make_node('MatMul',['query_rope','rt'],['rope_logits']),
        h.make_node('Add',['nope_logits','rope_logits'],['raw']),
        h.make_node('Mul',['raw','attention_scale'],['scaled']),
        h.make_node('Unsqueeze',['selected_valid','axis_h'],['mask']),
        h.make_node('Where',['mask','scaled','neg_inf'],['masked']),
        h.make_node('Cast',['selected_valid'],['valid_i64'],to=T.INT64),
        h.make_node('ReduceSum',['valid_i64','axis_k3'],['count'],keepdims=1),
        h.make_node('Greater',['count','zero_i64'],['any_valid3']),
        h.make_node('Unsqueeze',['any_valid3','axis_h'],['any_valid']),
        h.make_node('Where',['any_valid','masked','zero_float'],['safe_logits']),
        h.make_node('Softmax',['safe_logits'],['raw_weights'],axis=-1),
        h.make_node('Where',['mask','raw_weights','zero_float'],['weights']),
        h.make_node('MatMul',['weights','latent'],['latent_context']),
    ],[vi('latent',T.FLOAT,['B','S','K',512]),vi('rope_key',T.FLOAT,['B','S','K',64]),
       vi('query_absorbed',T.FLOAT,['B','S',64,512]),vi('query_rope',T.FLOAT,['B','S',64,64]),
       vi('selected_valid',T.BOOL,['B','S','K'])],
       [vi('latent_context',T.FLOAT,['B','S',64,512])],
       [const('axis_h',T.INT64,[2]),const('axis_k3',T.INT64,[2]),
        const('zero_i64',T.INT64,[0],[]),const('zero_float',T.FLOAT,[0.0],[]),
        const('neg_inf',T.FLOAT,[float('-inf')],[]),const('attention_scale',T.FLOAT,[256**-.5],[])])

if __name__=='__main__':
    build_glm()
    print('Built GLM-5.3 DSA and MLA ONNX graphs')
