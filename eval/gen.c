// sequentially generate k-nuggets---programs of length k exhibiting behavior
// not seen in any shorter program

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "bv8.h"

#define MAXLENGTH 9

// custom hash table for this
// quadratic probe
struct htab {
  int       ents; // number of entries
  char      *tab; // table
  uint64_t  *occ; // occupancy, bitpacked flags
};

// this hash is crap, figure out a better one later
// we'll run out of memory long before overflowing,
// so we can be careless about the mod
uint64_t hash(char *res, int len) {
  uint64_t h = 0;
  for (int i = 0; i < 256; i++) {
    h = (h * 33 + res[i]) % len;
  }
  return h;
}

// occupancy lookup logic
int hash_getocc(uint64_t *occ, int h) {
  return (occ[h / 64] >> (h % 64)) & 1;
}

void hash_setocc(uint64_t *occ, int h) {
  occ[h / 64] |= ((uint64_t) 1) << (h % 64);
}

char *hash_lookup(struct htab *t, char *res) {
  int ents = t->ents;
  char *table = t->tab;
  uint64_t *occ = t->occ;

  uint64_t h = hash(res, ents);

  // TODO more intelligent try-management?
  int inc = 1;
  for (int try = 0; try < 40; try++) {
    // check occupancy
    if (!hash_getocc(occ, h)) {
      goto fail;
    }

    // compare entry
    char *loc = &(table[h * 256]);
    for (int i = 0; i < 256; i++) {
      if (loc[i] != res[i]) {
        goto fail;
      }
    }

    return loc;

fail:
    h = (h + inc) % ents;
    inc++;
  }

  // fail!
  return NULL;
}

void hash_insert(struct htab *t, char *res) {
  int ents = t->ents;
  char *table = t->tab;
  uint64_t *occ = t->occ;

  uint64_t h = hash(res, ents);

  int inc = 1;
  for (int try = 0; try < 40; try++) {
    if (!hash_getocc(occ, h)) {
      // insert entry
      char *loc = &(table[h * 256]);
      for (int i = 0; i < 256; i++) {
        loc[i] = res[i];
      }
      hash_setocc(occ, h);
      return;
    } else {
      // duplicate check
      char *loc = &(table[h * 256]);
      for (int i = 0; i < 256; i++) {
        if (loc[i] != res[i]) {
          goto fail;
        }
      }
      return;
    }

fail:
    h = (h + inc) % ents;
    inc++;
  }

  // need to expand table...
  int new_ents = ents * 2;
  char *new_tab = malloc(256 * new_ents);
  uint64_t *new_occ = malloc(new_ents / 64 * sizeof(uint64_t));
  for (int i = 0; i < new_ents / 64; i++) {
    new_occ[i] = 0;
  }
  t->ents = new_ents;
  t->tab = new_tab;
  t->occ = new_occ;

  // copy over the old entries
  for (int i = 0; i < ents; i++) {
    if (hash_getocc(occ, i)) {
      char *loc = &(table[h * 256]);
      hash_insert(t, loc);
    }
  }
  free(table);
  free(occ);
  hash_insert(t, res); // and the new one
}

void hash_init(struct htab *t) {
  t->ents = 128;
  t->tab = malloc(128 * 256);
  t->occ = malloc(128 / 64 * sizeof(uint64_t));
}

// table per size
struct htab sz[MAXLENGTH] = {0};

// check a program output for duplication
int is_dup(char *res, int maxlen) {
  for (int i = 0; i < maxlen; i++) {
    if (hash_lookup(&sz[i], res)) {
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
    hash_init(&sz[i]);
  }

  // start generating programs
  char res[256] = {0};
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
        hash_insert(&sz[length - 1], res);
      }
      printf("\n");

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

    free(prog);
  }
}
