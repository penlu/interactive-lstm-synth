#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "bv8.h"
#include "sess.h"
#include "rand.h"
#include "giveup.h"

// fail fast cause we suck
int getchare() {
  int c = getchar();
  if (c == EOF) giveup(0);
  return c;
}

int readint() {
  int a = 0;
  int c;
  for (int i = 0; i < 4; i++) {
    c = getchare();
    a = (a << 8) | c;
  }

  return a;
}

int readprog(char *buf, int len) {
  for (int i = 0; i < len; i++) {
    buf[i] = getchare();
    if (buf[i] == 0) {
      return i;
    }
  }

  // too long
  giveup(0);
}

int main(int argc, char **argv) {
  // running in interactive mode by default

  bv8_init();

  // main interaction loop
  while (1) {
    char    prog_buf[2048];
    int     sess_id;        // session ID
    uint8_t **sess_vec;     // session behavior vector

    fflush(stdout);

    char c = getchare();

    // SESSION OPEN
    if (c == 1) {
      // desired session ID
      sess_id = readint();

      c = getchare(); // should be STX

      // get target program
      readprog(prog_buf, sizeof(prog_buf));

      // look up session data pointer
      // (store target behavior vec)
      sess_vec = sess_lookup(sess_id);
      if (*sess_vec != NULL) {
        free(*sess_vec);
      }
      *sess_vec = malloc(256);
      if (*sess_vec == NULL) giveup(0);

      // evaluate target program; result stored in session vector storage structure
      int err = bv8_eval(prog_buf, *sess_vec);
      if (err) giveup(0); // target progs shouldn't be messing up

    // SESSION QUERY
    } else if (c == 2) {
      // desired session ID
      sess_id = readint();

      c = getchare(); // should be NUL

      sess_vec = sess_lookup(sess_id);
      for (int i = 0; i < 256; i++) {
        putchar((*sess_vec)[i]);
      }
      putchar(0);

    // CANDIDATE QUERY
    } else if (c == 3) {
      // desired session ID
      sess_id = readint();

      c = getchare(); // should be STX
      
      // get target program
      readprog(prog_buf, sizeof(prog_buf));

      // evaluate candidate program
      uint8_t res[256];
      int err = bv8_eval(prog_buf, res);
      if (err == -1) {
        putchar('!');
        putchar(res[0]);
        putchar(0);
        continue;
      }
      if (err == -2) {
        putchar('?');
        putchar(res[0]);
        putchar(0);
        continue;
      }

      // look up session data pointer
      // (correct answers)
      sess_vec = sess_lookup(sess_id);

      // compare to session vector storage structure
      // store erroneous output
      // count number of bits correct!
      int     err_num = 0;
      uint8_t err_out[768];
      uint16_t wrong = 0;
      for (int i = 0; i < 256; i++) {
        if ((*sess_vec)[i] != res[i]) {
          err_out[err_num++] = i;
          err_out[err_num++] = (*sess_vec)[i];
          err_out[err_num++] = res[i];
        }

        uint8_t v = ((*sess_vec)[i] ^ res[i]);
        wrong += (v * 0x200040008001ULL & 0x111111111111111ULL) % 0xf;
      }

      if (err_num == 0) {
        putchar(0);
        continue;
      }

      // output all errors
      putchar('#');
      putchar((uint8_t) (err_num / 3));
      for (int i = 0; i < err_num; i++) {
        putchar(err_out[i]);
      }

      // output bitset count
      putchar(wrong >> 8);
      putchar(wrong & 0xff);

      putchar(0);
    }
  }
}
