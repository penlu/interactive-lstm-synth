// sequentially generate k-nuggets---programs of length k exhibiting behavior
// not seen in any shorter program

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "bv8.h"

#define MAXLENGTH 9

int main(int argc, char **argv) {
  // initialize evaluator
  bv8_init();

  // initialize hash tables
  for (int i = 0; i < MAXLENGTH; i++) {
    hash_init(&sz[i]);
  }

  // start generating programs
  uint8_t res[256] = {0};
  for (int length = 1; length <= MAXLENGTH; length++) {
    // make space for this size
    char *prog = malloc(length + 1);
    for (int i = 0; i < length; i++) {
      prog[i] = '@';
    }
    prog[length] = 0;

    while (1) {
      // evaluate current program
      int c = bv8_eval(prog, res);
      if (c != 0) { // invalid prog
        goto next;
      }

      // next program
next:
      {
        int i;
        for (i = 0; i < length; i++) {
          if (++prog[i] < 'M') {
            break;
          }
          prog[i] = '@';
        }
        if (i == length) {
          break;
        }
      }
    }
    printf("next\n");

    free(prog);
  }

  printf("done!\n");
  fflush(stdout);
}
