#!/usr/bin/env python

import random
import time

from eval import Evaluator

test = Evaluator()
test.sess_open(0, "B")
print test.sess_query(0)

print test.cand_query(0, "BBBJJ")
print test.cand_query(0, "BBBJ")
print test.cand_query(0, "J")

N = 10000

progs = []
for i in range(N):
  progs += [raw_input()]

randos = random.sample(range(100000000), N)
order = random.sample(range(N), N)

print "go"

start = time.time()

for i in range(N):
  test.sess_open(randos[i], progs[i])

opened = time.time() - start

for i in order:
  test.cand_query(randos[i], progs[i])

done = time.time() - start

print "open after "
print opened
print "done after "
print done

print test.read_odom()
