#include <stdlib.h>
#include <inttypes.h>
#include "bv8.h"
#include "giveup.h"

int stacksize = 0;
uint8_t *stack = NULL;

// stack element inc/dec by instruction
int inst_argc[13] = { 1, 1, 1,
                      0, 0, 0, 0, 0,
                      -1, -1, -1, -1,
                      -2 };

void bv8_init() {
  stack = malloc(64);
  if (stack == NULL) giveup(0);
  stacksize = 64;
}

int bv8_eval(char *prog, char *res) {
  // check for underflow/too many elements at termination
  int s = 0;
  for (int i = 0; prog[i]; i++) {
    s += inst_argc[prog[i] - '@'];

    if (s <= 0) {
      res[0] = i;
      return -1;
    }

    while (s >= stacksize) {
      stacksize *= 2;
      stack = realloc(stack, stacksize);
      if (stack == NULL) giveup(0);
    }
  }
  if (s != 1) {
    res[0] = s;
    return -2;
  }

  // evaluate on all inputs
  for (int x = 0; x < 256; x++) {
    int s = 0;
    for (int i = 0; prog[i]; i++) {
      char a, b, c;
      switch (prog[i]) {
        case 64:
          stack[s++] = 0;
          break;
        case 65:
          stack[s++] = 1;
          break;
        case 66:
          stack[s++] = x;
          break;
        case 67:
          stack[s - 1] = ~stack[s - 1];
          break;
        case 68:
          stack[s - 1] <<= 1;
          break;
        case 69:
          stack[s - 1] >>= 1;
          break;
        case 70:
          stack[s - 1] >>= 2;
          break;
        case 71:
          stack[s - 1] >>= 4;
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
          stack[s++] = a + b;
          break;
        case 76:
          a = stack[--s];
          b = stack[--s];
          c = stack[--s];
          stack[s++] = !a ? b : c;
          break;
      }
    }

    res[x] = stack[0];
  }

  return 0;
}
