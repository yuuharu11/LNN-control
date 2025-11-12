#!/bin/bash
#  task: square
#    dataset type: ph
#      hdf5 type: low_dim
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/low_dim/bc.json
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/low_dim/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/low_dim/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/low_dim/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/low_dim/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/low_dim/iris.json

"""
#  task: square
#    dataset type: ph
#      hdf5 type: image
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/image/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/image/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/image/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/ph/image/cql.json
"""

#  task: square
#    dataset type: mh
#      hdf5 type: low_dim
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/low_dim/bc.json
#python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/low_dim/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/low_dim/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/low_dim/cql.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/low_dim/hbc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/low_dim/iris.json

"""
#  task: square
#    dataset type: mh
#      hdf5 type: image
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/image/bc.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/image/bc_rnn.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/image/bcq.json
python /work/robomimic/robomimic/scripts/train.py --config /work/robomimic/robomimic/exps/paper/core/square/mh/image/cql.json
"""
