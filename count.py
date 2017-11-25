#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

l = [3]
counts = []
for c in range(10):
  l2 = [0] * (len(l) + 1)
  for i in range(len(l) + 1):
    l2[i] = 0
    if i - 1 >= 0:
      l2[i] += l[i - 1] * 3
    if i < len(l):
      l2[i] += l[i] * 5
    if i + 1 < len(l):
      l2[i] += l[i + 1] * 4
    if i + 2 < len(l):
      l2[i] += l[i + 2]
  print l
  counts += [math.log(l[0])]
  l = l2

regr = linear_model.LinearRegression()
regr.fit(np.stack([range(len(counts)), np.ones(len(counts))], axis=1), counts)
print regr.coef_, math.exp(regr.coef_[0])

plt.plot(range(len(counts)), counts, ".")
plt.plot(range(len(counts)), [i * math.log(13) for i in range(len(counts))], ".")
plt.show()
