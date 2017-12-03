// sequentially generate k-nuggets---programs of length k exhibiting behavior
// not seen in any shorter program

// we generate all programs and mark duplicates

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "bv8.h"
#include "hash.h"

#define MAXLENGTH 8

// table per size
struct htab *sz[MAXLENGTH] = {0};

// check a program output for duplication
int is_dup(uint8_t *res, int maxlen) {
  for (int i = 0; i < maxlen; i++) {
    if (hash_lookup(sz[i], res)) {
      return i + 1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  // initialize evaluator
  bv8_init();

  // initialize hash tables
  for (int i = 0; i < MAXLENGTH; i++) {
    sz[i] = hash_init();
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

      // output program and its output
      printf("%s ", prog);
      for (int i = 0; i < 256; i++) {
        printf("%02x", (uint8_t) res[i]);
      }

      // check prog output
      int duplen = is_dup(res, length);
      if (duplen) {
        printf(" dup %d", duplen);
      } else {
        hash_insert(sz[length - 1], res);
      }
      printf("\n");
      fflush(stdout);

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
