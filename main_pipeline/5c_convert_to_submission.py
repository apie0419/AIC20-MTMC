import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(BASE_PATH, "submission_adpt")
out_path = os.path.join(BASE_PATH, "track3.txt")

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