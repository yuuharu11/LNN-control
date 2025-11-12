
#  task: transport
#    dataset type: ph
#      hdf5 type: low_dim
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/low_dim/bc.json
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/low_dim/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/low_dim/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/low_dim/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/low_dim/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/low_dim/iris.json

"""
#  task: transport
#    dataset type: ph
#      hdf5 type: image
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/image/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/image/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/image/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/ph/image/cql.json
"""

#  task: transport
#    dataset type: mh
#      hdf5 type: low_dim
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/low_dim/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/low_dim/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/low_dim/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/low_dim/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/low_dim/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/low_dim/iris.json

"""
#  task: transport
#    dataset type: mh
#      hdf5 type: image
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/image/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/image/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/image/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/transport/mh/image/cql.json
"""