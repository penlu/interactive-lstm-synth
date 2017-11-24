#include <stdio.h>
#include <stdlib.h>

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
    char   prog_buf[2048];
    int    sess_id;      // session ID
    char **sess_vec;  // session behavior vector

    char c = getchare();

    // SESSION OPEN
    if (c == 1) {
      // desired session ID
      sess_id = readint();

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

      sess_vec = sess_lookup(sess_id);
      for (int i = 0; i < 256; i++) {
        putchar((*sess_vec)[i]);
      }
      putchar(0);

    // CANDIDATE QUERY
    } else if (c == 3) {
      // desired session ID
      sess_id = readint();
      
      // get target program
      readprog(prog_buf, sizeof(prog_buf));

      // evaluate candidate program
      char res[256];
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
      int  err_num = 0;
      char err_out[256];
      for (int i = 0; i < 256; i++) {
        if ((*sess_vec)[i] != res[i]) {
          err_out[err_num++] = i;
        }
      }

      if (err_num != 0) {
        // randomly select an erroneous output
        int i = rand_uniform(err_num);
        putchar('#');
        putchar(i);
        putchar((*sess_vec)[i]);
        putchar(res[i]);
      }

      // in any case...
      putchar(0);
    }
  }
}
