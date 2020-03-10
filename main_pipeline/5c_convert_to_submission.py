# -*- coding: utf-8 -*-
# author: peilun
# 转为提交格式
"""
"""

input_path = "/home/apie/AIC20_track3/mtmc-vt/src/submission_adpt"
out_path = "/home/apie/AIC20_track3/mtmc-vt/src/track1.txt"

f = open(out_path, 'w')
lines = open(input_path).readlines()
for line in lines:
    words = line.strip('\n').split(',')
    ww = str(int(words[0][2:]))
    for i in words[1:]:
        ww += ' ' + i
    ww += '\n'
    f.write(ww)

f.close()