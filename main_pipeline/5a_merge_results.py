# -*- coding: utf-8 -*-
import numpy as np
import os, sys

sys.path.append("..")

from config import cfg

input1 = os.path.join(cfg.PATH.ROOT_PATH, "submission_crossroad_train")
input2 = os.path.join(cfg.PATH.ROOT_PATH, "submission_normal_train")
out = os.path.join(cfg.PATH.ROOT_PATH, "submission")

f = open(out, 'w')

lines = open(input1).readlines()
for line in lines:
    f.write(line)

lines = open(input2).readlines()
for line in lines:
    f.write(line)

f.close()