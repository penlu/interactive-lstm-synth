all: eval gen

eval: eval.c bv8.c sess.c rand.c giveup.c
	gcc --std=c11 -Ofast -Werror -o eval eval.c bv8.c sess.c rand.c giveup.c

gen: gen.c bv8.c giveup.c
	gcc --std=c11 -O0 -g -Werror -o gen gen.c bv8.c sess.c rand.c giveup.c
