# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
import os
import glob
import shutil
import time


input1 = "/home/apie/AIC20_track3/mtmc-vt/src/submission_crossroad_train"
input2 = "/home/apie/AIC20_track3/mtmc-vt/src/submission_normal_train"
out = "/home/apie/AIC20_track3/mtmc-vt/src/submission"


f = open(out, 'w')

lines = open(input1).readlines()
for line in lines:
    f.write(line)

lines = open(input2).readlines()
for line in lines:
    f.write(line)

f.close()