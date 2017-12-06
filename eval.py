#!/usr/bin/env python
import os
import random
import struct
import time

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

# python binding for our evaluator friend!
# creates evaluator server subprocess, attaches to it with pair of pipes

def _child(fr_parent, to_parent):
  os.dup2(fr_parent, 0)
  os.dup2(to_parent, 1)
  os.close(fr_parent)
  os.close(to_parent)
  os.execl("/home/penlu/Documents/scratch/interactive-lstm-synth/eval/eval", "eval")

class Evaluator:

  def __init__(self):
    # fork off evaluator process
    pipe1r, pipe1w = os.pipe()
    pipe2r, pipe2w = os.pipe()

    pid = os.fork()
    if pid != 0:
      os.close(pipe1r)
      os.close(pipe2w)

      self.to_eval = pipe1w
      self.fr_eval = pipe2r

      self.sesscount = 0
      self.odometer = 0
      self.tottime = 0.

    else:
      os.close(pipe1w)
      os.close(pipe2r)

      _child(pipe1r, pipe2w)

  def sess_open(self, ID, prog):
    NUL = chr(0)
    SOH = chr(1)
    STX = chr(2)

    msg = SOH + struct.pack("<I", ID) + STX + prog + NUL
    os.write(self.to_eval, msg)

  def sess_query(self, ID):
    NUL = chr(0)
    STX = chr(2)

    msg = STX + struct.pack("<I", ID) + NUL
    start = time.time()
    os.write(self.to_eval, msg)

    resp = os.read(self.fr_eval, 256)
    self.tottime += time.time() - start
    self.odometer += 1
    assert len(resp) == 256
    assert os.read(self.fr_eval, 1) == NUL
    return [ord(c) for c in list(resp)]

  def cand_query(self, ID, prog):
    NUL = chr(0)
    STX = chr(2)
    ETX = chr(3)

    msg = ETX + struct.pack("<I", ID) + STX + prog + NUL
    start = time.time()
    os.write(self.to_eval, msg)

    # read result character
    res = os.read(self.fr_eval, 1)
    self.tottime += time.time() - start
    self.odometer += 1

    if res == '!':
      # stack underflow during execution, at position...
      pos = ord(os.read(self.fr_eval, 1))
      assert os.read(self.fr_eval, 1) == NUL
      #return ('!', pos)
      retval = ('!', pos)
      return (-30., torch.LongTensor([[20], [int(pos)/8+2], [int(pos)%8+2]]))
    elif res == '?':
      # stack overflow at termination, count...
      cnt = ord(os.read(self.fr_eval, 1))
      assert os.read(self.fr_eval, 1) == NUL
      #return ('?', cnt)
      retval = ('?', cnt)
      return (-30., torch.LongTensor([[21], [int(cnt)/8+2], [int(pos)%8+2]]))
    elif res == '#':
      # incorrect outputs
      cnt = ord(os.read(self.fr_eval, 1))
      #print cnt
      # handle overflows on remote end
      if cnt == 0:
        cnt = 256
      resp = os.read(self.fr_eval, cnt * 3)
      wrong = ord(os.read(self.fr_eval, 1))*256 + ord(os.read(self.fr_eval, 1))

      # list of incorrect results
      assert os.read(self.fr_eval, 1) == NUL
      #return ('#', [(ord(resp[cnt * 3]), ord(resp[cnt * 3 + 1]), ord(resp[cnt * 3 + 2])) for i in range(cnt)], wrong)

      samp = random.randint(0, cnt - 1)
      # input, correct, incorrect
      retval = (ord(resp[samp * 3]), ord(resp[samp * 3 + 1]), ord(resp[samp * 3 + 2]))
      tenret = torch.LongTensor([[retval[0]/8+2], [retval[0]%8+2], [18],
                                 [retval[1]/8+2], [retval[1]%8+2], [18],
                                 [retval[2]/8+2], [retval[2]%8+2], [19]])
      return (float(2048 - wrong) / 80 - 30, tenret)
    elif res == NUL:
      # no errors, we're done!
      #return ()
      return (100000., torch.LongTensor([[1]]))

  def _candquery(self, ID):
    def inner(prog):
      print prog
      return self.cand_query(ID, prog)

    return inner

  def eval_init(self, prog):
    print prog
    self.sess_open(self.sesscount, prog)
    self.sesscount += 1

    return self._candquery(self.sesscount - 1)

  def read_odom(self):
    avg = self.tottime / self.odometer
    print self.odometer
    print self.tottime
    self.tottime = 0
    self.odometer = 0
    return avg

