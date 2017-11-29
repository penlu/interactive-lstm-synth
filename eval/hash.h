
struct htab;

uint8_t *hash_lookup(struct htab *t, uint8_t *res);
void hash_insert(struct htab *t, uint8_t *res);
struct htab *hash_init(void);
