# -*- coding: UTF-8 -*-
import numpy as np
from tqdm import tqdm
import sys


keys = "$abcdefghijklmnopqrstuvwxyzABCDEGHIJKLMNOPQRSTUVZ#"

lst_plus = []
lstpath=sys.argv[1]
#eg:F:\\seq2seq\\doubletrain_doubletest\\4folds_cross_validation\\testing_set_fold1_of_4cv_label.lst
with open(lstpath) as f:
    lst = f.readlines()

for i in lst:
    i = str(i).replace("\n","")
    i += "#"
    lst_plus.append(i)

X = np.zeros((len(lst_plus), 25, 1), dtype='int32')

length = 0

zz = "P"
xx = ""

for i in tqdm(range(len(lst_plus))):
    t = []
    l = lst_plus[i]
    l = l.strip()

    try:
        xx = l
        for k in l:
            zz = k
            a = keys.index(k)
            t.append(a)
        #t = [keys.index(k) for k in l]
    except ValueError as e:
        print(e)
        print(zz)
        print(xx)
        #raise

    for j in range(len(t)):
        X[i, j, 0] = t[j]


np.save(lstpath[:-4], X)
