#include <stdlib.h>

int rand_uniform(int a) {
  int range = (long long) RAND_MAX - (((long long) RAND_MAX + 1) % (a + 1));
  int slotsize = ((long long) range + 1) / (a + 1);
  while (1) {
    int r = rand();
    if (r > range) {
      continue;
    } else {
      return r / slotsize;
    }
  }
}
