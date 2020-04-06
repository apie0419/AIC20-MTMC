import os, sys

sys.path.append("..")

from config import cfg

input_path = os.path.join(cfg.PATH.ROOT_PATH, "submission_adpt")
out_path = os.path.join(cfg.PATH.ROOT_PATH, "track3.txt")

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