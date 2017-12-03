#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "bv8.h"
#include "giveup.h"

#include "immintrin.h"

// this is way more than enough
#define STACKSIZE 64

// store whole groups of 32-byte values
__m256i stack[STACKSIZE+2] __attribute__ ((aligned (32)));

// stack element inc/dec by instruction
int inst_argc[13] = { 1, 1, 1,
                      0, 0, 0, 0, 0,
                      -1, -1, -1, -1,
                      -2 };

void bv8_init() {
  // I guess we could check if AVX-2...
  return;
}

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

// stack caching trickery: 8-register hand-over-hand
// we maintain a cache of the stack with at least 2 valid slots
// below any given position:
/*
  BLOCKr0:
        v
  r6 r7 r0 r1 r2 r3 r4 r5

  BLOCKr1:
           v
  r6 r7 r0 r1 r2 r3 r4 r5
        x

  BLOCKr2:
              v
  r6 r7 r0 r1 r2 r3 r4 r5
        x  x

  BLOCKr3:
                 v
  r6 r7 r0 r1 r2 r3 r4 r5
        x  x  x

  BLOCKr4:
                    v
              r2 r3 r4 r5 r6 r7 r0 r1
        x  x  x  x

  BLOCKr5:
                       v
              r2 r3 r4 r5 r6 r7 r0 r1
        x  x  x  x  x

  BLOCKr6:
                          v
              r2 r3 r4 r5 r6 r7 r0 r1
        x  x  x  x  x  x

  BLOCKr7:
                             v
              r2 r3 r4 r5 r6 r7 r0 r1
        x  x  x  x  x  x  x

  BLOCKr0:
                                v
                          r6 r7 r0 r1 r2 r3 r4 r5
        x  x  x  x  x  x  x  x

  BLOCKr1:
                                   v
                          r6 r7 r0 r1 r2 r3 r4 r5
        x  x  x  x  x  x  x  x  x
*/
// control flow is simplified because our movement is purely
// register-dependent
// we take memory hits when we cross the r7/r0 and r3/r4 line

// in the ensuing, pass in the true alignment of cache reg 0

// shuffle cache: write out + reload r2-r5
// occurs when we pass r7 -> r0
#define LOCACHE_UP(s) \
  sstack[s - 6] = r2; \
  sstack[s - 5] = r3; \
  sstack[s - 4] = r4; \
  sstack[s - 3] = r5; \
  r2 = sstack[s + 2]; \
  r3 = sstack[s + 3]; \
  r4 = sstack[s + 4]; \
  r5 = sstack[s + 5];

// occurs when we pass r0 -> r7
#define LOCACHE_DN(s) \
  sstack[s + 2] = r2; \
  sstack[s + 3] = r3; \
  sstack[s + 4] = r4; \
  sstack[s + 5] = r5; \
  r2 = sstack[s - 6]; \
  r3 = sstack[s - 5]; \
  r4 = sstack[s - 4]; \
  r5 = sstack[s - 3];

// shuffle cache: write out + reload r6-r1
// occurs when we pass r3 -> r4
#define HICACHE_UP(s) \
  sstack[s - 2] = r6; \
  sstack[s - 1] = r7; \
  sstack[s + 0] = r0; \
  sstack[s + 1] = r1; \
  r6 = sstack[s + 6]; \
  r7 = sstack[s + 7]; \
  r0 = sstack[s + 8]; \
  r1 = sstack[s + 9];

// occurs when we pass r4 -> r3
#define HICACHE_DN(s) \
  sstack[s + 6] = r6; \
  sstack[s + 7] = r7; \
  sstack[s + 8] = r0; \
  sstack[s + 9] = r1; \
  r6 = sstack[s - 2]; \
  r7 = sstack[s - 1]; \
  r0 = sstack[s + 0]; \
  r1 = sstack[s + 1];

#define REGBLOCK(r_at1, r_at0, r_atn1, r_atn2, r_atn3, UP1, DOWN1, DOWN2) \
  while (prog[i]) { \
    switch (prog[i]) { \
      __m256i a; \
      case 64: \
        UP1; \
        r_at0 = zero; \
        s++; \
        goto block##r_at1; \
      case 65: \
        UP1; \
        r_at0 = one; \
        s++; \
        goto block##r_at1; \
      case 66: \
        UP1; \
        r_at0 = vecs[v]; \
        s++; \
        goto block##r_at1; \
      case 67: \
        r_atn1 = ~r_atn1; \
        goto block##r_at0; \
      case 68: \
        r_atn1 = (r_atn1 << 1) & mask_fe; \
        goto block##r_at0; \
      case 69: \
        r_atn1 = (r_atn1 >> 1) & mask_7f; \
        goto block##r_at0; \
      case 70: \
        r_atn1 = (r_atn1 >> 2) & mask_3f; \
        goto block##r_at0; \
      case 71: \
        r_atn1 = (r_atn1 >> 4) & mask_0f; \
        goto block##r_at0; \
      case 72: \
        DOWN1; \
        r_atn2 = r_atn2 & r_atn1; \
        s--; \
        goto block##r_atn1; \
      case 73: \
        DOWN1; \
        r_atn2 = r_atn2 | r_atn1; \
        s--; \
        goto block##r_atn1; \
      case 74: \
        DOWN1; \
        r_atn2 = r_atn2 ^ r_atn1; \
        s--; \
        goto block##r_atn1; \
      case 75: \
        DOWN1; \
        r_atn2 = _mm256_add_epi8(r_atn2, r_atn1); \
        s--; \
        goto block##r_atn1; \
      case 76: \
        DOWN2; \
        a = _mm256_cmpeq_epi8(r_atn1, zero); \
        r_atn3 = (a & r_atn2) | ((~ a) & r_atn3); \
        s -= 2; \
        goto block##r_atn2; \
    } \
block##r_at0: \
    i++; \
  }

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

  __m256i *sstack = stack + 2;

  // evaluate on all inputs
  // we use AVX-2 GCC intrinsics for this, evaluating in data-parallel
  // on 8 groups of 32 byte values (256 bits) each
  for (int v = 0; v < 8; v++) {
    int s = 0;

    __m256i r0, r1, r2, r3, r4, r5, r6, r7; // stack cache

    int i = 0;
    REGBLOCK(r1, r0, r7, r6, r5, , LOCACHE_DN(s - 0), LOCACHE_DN(s - 0));
    REGBLOCK(r2, r1, r0, r7, r6, , , LOCACHE_DN(s - 1));
    REGBLOCK(r3, r2, r1, r0, r7, , , );
    REGBLOCK(r4, r3, r2, r1, r0, HICACHE_UP(s - 3), , );
    REGBLOCK(r5, r4, r3, r2, r1, , HICACHE_DN(s - 4), HICACHE_DN(s - 4));
    REGBLOCK(r6, r5, r4, r3, r2, , , HICACHE_DN(s - 5));
    REGBLOCK(r7, r6, r5, r4, r3, , , );
    REGBLOCK(r0, r7, r6, r5, r4, LOCACHE_UP(s + 1), , );

    // copy result out
    // should end with result in r0
    sstack[0] = r0;
    for (int i = 0; i < 32; i++) {
      res[v * 32 + i] = ((uint8_t*) (&sstack[0]))[i];
    }
  }

  return 0;
}
