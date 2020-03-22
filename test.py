class T(object):
    def __init__(self, _id):
        self.id = _id

t1 = T(1)
t2 = T(2)
t3 = T(3)

allt = {"test": [t1, t2, t3]}
tl = allt["test"]

def test(tl):
    for t in tl:
        t.id = 5

test(tl)
for t in allt["test"]:
    print (t.id)
