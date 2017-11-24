#!/usr/bin/env python
import os
import struct

# python binding for our evaluator friend!
# creates evaluator server subprocess, attaches to it with pair of pipes

def _child(fr_parent, to_parent):
  os.dup2(fr_parent, 0)
  os.dup2(to_parent, 1)
  os.close(fr_parent)
  os.close(to_parent)
  os.execl("/home/penlu/Documents/scratch/interactive-lstm-synth/eval/eval")

class Evaluator:

  def __init__(self, serialize):
    # fork off evaluator process
    pipe1r, pipe1w = os.pipe()
    pipe2r, pipe2w = os.pipe()

    pid = os.fork()
    if pid == 0:
      os.close(pipe1r)
      os.close(pipe2w)

      self.to_eval = pipe1w
      self.fr_eval = pipe2r

    else:
      os.close(pipe1w)
      os.close(pipe2r)

      _child(pipe1r, pipe2w)

  def sess_open(ID, prog):
    NUL = chr(0)
    SOH = chr(1)
    STX = chr(2)

    msg = SOH + struct.pack("<I", ID) + STX + self.serialize(prog) + NUL
    os.write(self.to_eval, msg)

  def sess_query(ID):
    NUL = chr(0)
    STX = chr(2)

    msg = STX + struct.pack("<I", ID) + NUL
    os.write(self.to_eval, msg)

    resp = os.read(self.from_eval, 256)
    assert len(resp) == 256
    assert os.read(self.from_eval, 1) == NUL
    return [ord(c) for c in list(resp)]

  def cand_query(ID, prog):
    NUL = chr(0)
    STX = chr(2)
    ETX = chr(3)

    msg = ETX + struct.pack("<I", ID) + self.serialize(prog) + NUL

    res = os.read(self.from_eval, 1)
    # TODO process response
