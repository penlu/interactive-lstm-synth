#include <inttypes.h>
// session data structure is a map from sess_id -> char*, intended as
// behavior vectors

// gives a pointer to a slot in the session data structure for given session ID
uint8_t **sess_lookup(int sess_id);
