// hash elements are 256 bytes only!

#include <stdlib.h>
#include <inttypes.h>

#include "hash.h"

// this hash is crap, figure out a better one later
// we'll run out of memory long before overflowing,
// so we can be careless about the mod
uint64_t hash(uint8_t *res, uint64_t len) {
  uint64_t h = 0;
  for (int i = 0; i < 256; i++) {
    h = (h * 33 + res[i]) % len;
  }
  return h;
}

// occupancy lookup logic
int hash_getocc(uint64_t *occ, uint64_t h) {
  return (occ[h / 64] >> (h % 64)) & 1;
}

void hash_setocc(uint64_t *occ, uint64_t h) {
  occ[h / 64] |= ((uint64_t) 1) << (h % 64);
}

uint8_t *hash_lookup(struct htab *t, uint8_t *res) {
  size_t ents = t->ents;
  uint8_t *table = t->tab;
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
    uint8_t *loc = &(table[h * 256]);
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

void hash_insert(struct htab *t, uint8_t *res) {
  size_t ents = t->ents;
  uint8_t *table = t->tab;
  uint64_t *occ = t->occ;

  uint64_t h = hash(res, ents);

  int inc = 1;
  for (int try = 0; try < 40; try++) {
    if (!hash_getocc(occ, h)) {
      // insert entry
      uint8_t *loc = &(table[h * 256]);
      for (int i = 0; i < 256; i++) {
        loc[i] = res[i];
      }
      hash_setocc(occ, h);
      return;
    } else {
      // duplicate check
      uint8_t *loc = &(table[h * 256]);
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
  size_t new_ents = ents * 2;
  uint8_t *new_tab = calloc(new_ents, 256);
  uint64_t *new_occ = calloc(new_ents / 64, sizeof(uint64_t));
  for (int i = 0; i < new_ents / 64; i++) {
    new_occ[i] = 0;
  }
  t->ents = new_ents;
  t->tab = new_tab;
  t->occ = new_occ;

  // copy over the old entries
  for (size_t i = 0; i < ents; i++) {
    if (hash_getocc(occ, i)) {
      uint8_t *loc = &(table[h * 256]);
      hash_insert(t, loc);
    }
  }
  free(table);
  free(occ);
  hash_insert(t, res); // and the new one
}

#define HASHINITSIZE 128
void hash_init(struct htab *t) {
  t->ents = HASHINITSIZE;
  t->tab = calloc(HASHINITSIZE, 256);
  t->occ = calloc(HASHINITSIZE / 64, sizeof(uint64_t));
}

