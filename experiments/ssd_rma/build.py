"""Build dimension-independent, FP32 ONNX contracts. No trained weights included."""
from pathlib import Path
import onnx
from onnx import helper as h, TensorProto as T

ROOT = Path(__file__).resolve().parent

def vi(name, dtype, shape):
    return h.make_tensor_value_info(name, dtype, shape)

def const(name, dtype, values, shape=None):
    return h.make_tensor(name, dtype, [len(values)] if shape is None else shape, values)

def save(name, nodes, inputs, outputs, initializers):
    graph = h.make_graph(nodes, name, inputs, outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 18)],
                         producer_name='edgerock-storage-poc')
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, ROOT / (name + '.onnx'))

def build():
    # u is AFTER EdgeRock input_norm; both projection matrices are explicit inputs.
    save('admission_score', [
        h.make_node('MatMul', ['u', 'down'], ['low']),
        h.make_node('Sigmoid', ['low'], ['sig']),
        h.make_node('Mul', ['low', 'sig'], ['features']),
        h.make_node('MatMul', ['features', 'up'], ['score_bng']),
        h.make_node('Transpose', ['score_bng'], ['scores'], perm=[0, 2, 1]),
    ], [vi('u',T.FLOAT,['B','N','H']), vi('down',T.FLOAT,['H','R']),
        vi('up',T.FLOAT,['R','G'])], [vi('scores',T.FLOAT,['B','G','N'])], [])

    # Sort by position ascending, then score descending. ONNX TopK breaks ties
    # with lower input index, preserving source select_topk's stable ordering.
    save('stable_select', [
        h.make_node('Less', ['references', 'zero'], ['invalid']),
        h.make_node('Where', ['invalid','max_i64','positions'], ['sort_positions']),
        h.make_node('Where', ['invalid','neg_inf','scores'], ['masked_scores']),
        h.make_node('Shape', ['scores'], ['shape']),
        h.make_node('Gather', ['shape','last_axis'], ['n'], axis=0),
        h.make_node('TopK', ['sort_positions','n'], ['unused_positions','position_order'], axis=-1,largest=0,sorted=1),
        h.make_node('GatherElements', ['masked_scores','position_order'], ['ordered_scores'],axis=-1),
        h.make_node('TopK', ['ordered_scores','capacity'], ['selected_scores','score_order'],axis=-1,largest=1,sorted=1),
        h.make_node('GatherElements', ['position_order','score_order'], ['selected_indices'],axis=-1),
        h.make_node('GatherElements', ['references','selected_indices'], ['selected_references'],axis=-1),
        h.make_node('GatherElements', ['positions','selected_indices'], ['raw_positions'],axis=-1),
        h.make_node('Less', ['selected_references','zero'], ['selected_invalid']),
        h.make_node('Where', ['selected_invalid','minus_one','raw_positions'], ['selected_positions']),
    ], [vi('scores',T.FLOAT,['B','G','N']),vi('positions',T.INT64,['B','G','N']),
        vi('references',T.INT64,['B','G','N']),vi('capacity',T.INT64,[1])],
       [vi('selected_scores',T.FLOAT,['B','G','K']),vi('selected_positions',T.INT64,['B','G','K']),
        vi('selected_references',T.INT64,['B','G','K']),vi('selected_indices',T.INT64,['B','G','K'])],
       [const('zero',T.INT64,[0],[]), const('minus_one',T.INT64,[-1],[]),
        const('max_i64',T.INT64,[9223372036854775807],[]),
        const('last_axis',T.INT64,[2]),const('neg_inf',T.FLOAT,[float('-inf')],[])])

    # Memory read only: null entry reproduces EdgeRock's zero-logit null slot.
    # Q is number of query heads per KV group (single decode step).
    save('memory_read', [
        h.make_node('Transpose',['keys'],['kt'],perm=[0,1,3,2]),
        h.make_node('MatMul',['query','kt'],['raw_logits']),
        h.make_node('Mul',['raw_logits','scale'],['logits']),
        h.make_node('Unsqueeze',['valid','axis_q'],['mask']),
        h.make_node('Where',['mask','logits','neg_inf'],['masked']),
        h.make_node('Pad',['masked','pads','null_logit'],['with_null'],mode='constant'),
        h.make_node('Softmax',['with_null'],['all_weights'],axis=-1),
        h.make_node('Slice',['all_weights','start','end','axis_k'],['weights']),
        h.make_node('MatMul',['weights','values'],['output']),
    ],[vi('query',T.FLOAT,['B','G','Q','D']),vi('keys',T.FLOAT,['B','G','K','D']),
       vi('values',T.FLOAT,['B','G','K','D']),vi('valid',T.BOOL,['B','G','K']),
       vi('scale',T.FLOAT,[])], [vi('output',T.FLOAT,['B','G','Q','D'])],
       [const('axis_q',T.INT64,[2]),const('neg_inf',T.FLOAT,[float('-inf')],[]),
        const('null_logit',T.FLOAT,[0.0],[]),const('pads',T.INT64,[0,0,0,0,0,0,0,1]),
        const('start',T.INT64,[0]),const('end',T.INT64,[-1]),const('axis_k',T.INT64,[3])])

if __name__ == '__main__':
    build()
    print('Built and checked 3 ONNX graphs in', ROOT)
