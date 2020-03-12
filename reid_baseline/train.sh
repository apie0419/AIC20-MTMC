#! /bin/bash

nohup python tools/train.py --config_file="configs/softmax_triple.yml" 2>&1 &
