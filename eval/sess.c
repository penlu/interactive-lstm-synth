#include <stdlib.h>
#include <inttypes.h>
#include "sess.h"
#include "giveup.h"

uint8_t **sess_root[65536] = {NULL};

uint8_t **sess_lookup(int sess_id) {
  int i1 = (sess_id & 0xffff0000) >> 16;
  int i2 = (sess_id & 0x0000ffff);

  if (sess_root[i1] == NULL) {
    sess_root[i1] = malloc(65536 * sizeof(uint8_t*));
    if (sess_root[i1] == NULL) giveup(0);
    for (int i = 0; i < 65536; i++) {
      sess_root[i1][i] = 0;
    }
  }

  return &sess_root[i1][i2];
}
