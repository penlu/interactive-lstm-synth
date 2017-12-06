#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void giveup(int code) {
  fprintf(stderr, "GIVE UP!\n");
  fflush(stdout);
  fflush(stderr);
  /*while (1) {
    sleep(3);
  }*/
  exit(code);
}
