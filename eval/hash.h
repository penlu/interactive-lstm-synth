// hash elements are 256 bytes only!

// custom hash table for this
// quadratic probe
struct htab {
  size_t    ents; // number of entries
  uint8_t   *tab; // table
  uint64_t  *occ; // occupancy, bitpacked flags
};

uint8_t *hash_lookup(struct htab *t, uint8_t *res);
void hash_insert(struct htab *t, uint8_t *res);
void hash_init(struct htab *t);
