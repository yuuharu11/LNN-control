#!/bin/bash

# ==========core==========

#  task: lift
#    dataset type: ph
#      hdf5 type: low_dim
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/low_dim/bc.json
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/low_dim/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/low_dim/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/low_dim/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/low_dim/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/low_dim/iris.json

"""
#  task: lift
#    dataset type: ph
#      hdf5 type: image
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/image/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/image/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/image/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/ph/image/cql.json
"""

#  task: lift
#    dataset type: mh
#      hdf5 type: low_dim
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/low_dim/bc.json
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/low_dim/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/low_dim/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/low_dim/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/low_dim/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/low_dim/iris.json

"""
#  task: lift
#    dataset type: mh
#      hdf5 type: image
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/image/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/image/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/image/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mh/image/cql.json
"""
"""
#  task: lift
#    dataset type: mg
#      hdf5 type: low_dim_sparse
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_sparse/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_sparse/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_sparse/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_sparse/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_sparse/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_sparse/iris.json

#  task: lift
#    dataset type: mg
#      hdf5 type: image_sparse
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/image_sparse/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/image_sparse/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/image_sparse/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/image_sparse/cql.json

#  task: lift
#    dataset type: mg
#      hdf5 type: low_dim_dense
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_dense/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_dense/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_dense/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_dense/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_dense/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/low_dim_dense/iris.json

#  task: lift
#    dataset type: mg
#      hdf5 type: image_dense
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/image_dense/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/image_dense/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/image_dense/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/lift/mg/image_dense/cql.json
"""
