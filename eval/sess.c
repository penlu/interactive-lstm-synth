#include <stdlib.h>
#include "sess.h"

char **sess_root[65536] = {NULL};

char **sess_lookup(int sess_id) {
  int i1 = (sess_id & 0xffff0000) >> 16;
  int i2 = (sess_id & 0x0000ffff);

  if (sess_root[i1] == NULL) {
    sess_root[i1] = malloc(65536 * sizeof(char*));
    if (sess_root[i1] == NULL) exit(0);
    for (int i = 0; i < 65536; i++) {
      sess_root[i1][i] = 0;
    }
  }

  return &sess_root[i1][i2];
}
