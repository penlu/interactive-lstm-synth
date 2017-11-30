#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "bv8.h"
#include "giveup.h"

#include "immintrin.h"

// this is way more than enough
#define STACKSIZE 64

// store whole groups of 32-byte values
__m256i stack[STACKSIZE] __attribute__ ((aligned (32)));

// stack element inc/dec by instruction
int inst_argc[13] = { 1, 1, 1,
                      0, 0, 0, 0, 0,
                      -1, -1, -1, -1,
                      -2 };

void bv8_init() {
  // I guess we could check if AVX-2...
  return;
}

// trying constant array for now; may need to do it differently
const __m256i vecs[8] __attribute__ ((aligned (32))) = {
  { 0x0706050403020100, 0x0f0e0d0c0b0a0908,
    0x1716151413121110, 0x1f1e1d1c1b1a1918},
  { 0x2726252423222120, 0x2f2e2d2c2b2a2928,
    0x3736353433323130, 0x3f3e3d3c3b3a3938},
  { 0x4746454443424140, 0x4f4e4d4c4b4a4948,
    0x5756555453525150, 0x5f5e5d5c5b5a5958},
  { 0x6766656463626160, 0x6f6e6d6c6b6a6968,
    0x7776757473727170, 0x7f7e7d7c7b7a7978},
  { 0x8786858483828180, 0x8f8e8d8c8b8a8988,
    0x9796959493929190, 0x9f9e9d9c9b9a9998},
  { 0xa7a6a5a4a3a2a1a0, 0xafaeadacabaaa9a8,
    0xb7b6b5b4b3b2b1b0, 0xbfbebdbcbbbab9b8},
  { 0xc7c6c5c4c3c2c1c0, 0xcfcecdcccbcac9c8,
    0xd7d6d5d4d3d2d1d0, 0xdfdedddcdbdad9d8},
  { 0xe7e6e5e4e3e2e1e0, 0xefeeedecebeae9e8,
    0xf7f6f5f4f3f2f1f0, 0xfffefdfcfbfaf9f8}
};

// also wanted: special constants 0, 1, and shiftmasks
const __m256i zero =
  {0x0000000000000000, 0x0000000000000000,
   0x0000000000000000, 0x0000000000000000};

const __m256i one =
  {0x0101010101010101, 0x0101010101010101,
   0x0101010101010101, 0x0101010101010101};

const __m256i mask_ff =
  {0xffffffffffffffff, 0xffffffffffffffff,
   0xffffffffffffffff, 0xffffffffffffffff};

// for use when shifting left 1
const __m256i mask_fe =
  {0xfefefefefefefefe, 0xfefefefefefefefe,
   0xfefefefefefefefe, 0xfefefefefefefefe};

// for use when shifting right 1
const __m256i mask_7f =
  {0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f,
   0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};

// for use when shifting right 2
const __m256i mask_3f =
  {0x3f3f3f3f3f3f3f3f, 0x3f3f3f3f3f3f3f3f,
   0x3f3f3f3f3f3f3f3f, 0x3f3f3f3f3f3f3f3f};

// for use when shifting right 4
const __m256i mask_0f =
  {0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f,
   0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f};

int bv8_eval(char *prog, uint8_t *res) {
  // check for underflow/too many elements at termination
  int s = 0;
  for (int i = 0; prog[i]; i++) {
    s += inst_argc[prog[i] - '@'];

    if (s <= 0 || s >= STACKSIZE) {
      res[0] = i;
      return -1;
    }
  }
  if (s != 1) {
    res[0] = s;
    return -2;
  }

  // evaluate on all inputs
  // we use AVX-2 GCC intrinsics for this, evaluating in data-parallel
  // on 8 groups of 32 byte values (256 bits) each
  for (int v = 0; v < 8; v++) {
    int s = 0;
    for (int i = 0; prog[i]; i++) {
      __m256i a, b, c;
      switch (prog[i]) {
        case 64:
          stack[s++] = zero;
          break;
        case 65:
          stack[s++] = one;
          break;
        case 66:
          stack[s++] = vecs[v];
          break;
        case 67:
          stack[s - 1] = ~stack[s - 1];
          break;
        case 68:
          stack[s - 1] = (stack[s - 1] << 1) & mask_fe;
          break;
        case 69:
          stack[s - 1] = (stack[s - 1] >> 1) & mask_7f;
          break;
        case 70:
          stack[s - 1] = (stack[s - 1] >> 2) & mask_3f;
          break;
        case 71:
          stack[s - 1] = (stack[s - 1] >> 4) & mask_0f;
          break;
        case 72:
          a = stack[--s];
          b = stack[--s];
          stack[s++] = a & b;
          break;
        case 73:
          a = stack[--s];
          b = stack[--s];
          stack[s++] = a | b;
          break;
        case 74:
          a = stack[--s];
          b = stack[--s];
          stack[s++] = a ^ b;
          break;
        case 75:
          a = stack[--s];
          b = stack[--s];
          stack[s++] = _mm256_add_epi8(a, b);
          break;
        case 76:
          a = _mm256_cmpeq_epi8(stack[--s], zero);
          b = stack[--s];
          c = stack[--s];
          stack[s++] = (a & b) | ((~ a) & c);
          break;
      }
    }

    // copy result out
    for (int i = 0; i < 32; i++) {
      res[v * 32 + i] = ((uint8_t*) stack)[i];
    }
  }

  return 0;
}
