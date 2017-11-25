#include <inttypes.h>

// initialize the evaluation stack
void bv8_init(void);

// evaluate 8-bit bitvector program `prog`
// results are placed in 256-slot array `res`
// return value:
// 0 means successful evaluation
// -1 means the program underflows; res[0] has the inst no. where this occurs
// -2 means the program terminates with too many elements; res[0] has how many
int bv8_eval(char *prog, uint8_t *res);
