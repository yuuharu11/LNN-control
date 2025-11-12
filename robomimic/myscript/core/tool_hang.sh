#  task: tool_hang
#    dataset type: ph
#      hdf5 type: low_dim
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/low_dim/bc.json
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/low_dim/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/low_dim/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/low_dim/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/low_dim/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/low_dim/iris.json

"""
#  task: tool_hang
#    dataset type: ph
#      hdf5 type: image
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/image/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/image/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/image/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/tool_hang/ph/image/cql.json
"""