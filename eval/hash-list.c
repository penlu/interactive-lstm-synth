// hash elements are 256 bytes only!

#include <stdlib.h>
#include <inttypes.h>

#include "hash.h"

struct htab {
  size_t    ents; // number of entries
  size_t    ctr;  // number of items
  struct l  **tab; // table
};

struct l {
  struct l  *next;
  uint8_t   data[256];
};

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

uint8_t *hash_lookup(struct htab *t, uint8_t *res) {
  size_t ents = t->ents;
  struct l **table = t->tab;

  uint64_t h = hash(res, ents);

  struct l *head = table[h];
  while (head) {
    // compare entry
    for (int i = 0; i < 256; i++) {
      if (head->data[i] != res[i]) {
        goto fail;
      }
    }

    return head->data;

fail:
    head = head->next;
  }

  return NULL;
}

void hash_insert(struct htab *t, uint8_t *res) {
  size_t ents = t->ents;
  struct l **table = t->tab;

  // time to expand table?
  if (t->ctr >= ents * 8) {
    // allocate new entries
    size_t new_ents = ents * 2;
    struct l **new_tab = calloc(new_ents, sizeof(struct l *));

    t->ents = new_ents;
    t->tab = new_tab;

    // copy over the old entries
    for (size_t i = 0; i < ents; i++) {
      struct l *head = table[i];
      while (head) {
        struct l *next = head->next;

        uint64_t h = hash(head->data, new_ents);
        head->next = new_tab[h];
        new_tab[h] = head;

        head = next;
      }
    }

    free(table);
    ents = t->ents;
    table = t->tab;
  }

  // insert new entry
  uint64_t h = hash(res, ents);
  struct l *new = malloc(sizeof(struct l));
  for (int i = 0; i < 256; i++) {
    new->data[i] = res[i];
  }
  new->next = t->tab[h];
  t->tab[h] = new;

  t->ctr++;
}

#define HASHINITSIZE 128
struct htab *hash_init() {
  struct htab *t = malloc(sizeof(struct htab));
  t->ents = HASHINITSIZE;
  t->ctr = 0;
  t->tab = calloc(HASHINITSIZE, sizeof(struct l *));
  return t;
}

