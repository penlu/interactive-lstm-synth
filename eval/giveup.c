#include <stdio.h>
#include <stdlib.h>

void giveup(int code) {
  fprintf(stderr, "GIVE UP!\n");
  exit(code);
}
