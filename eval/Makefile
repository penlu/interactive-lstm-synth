all: eval gen gen2

eval: eval.c bv8.c sess.c rand.c giveup.c
	gcc --std=c11 -march=native -Ofast -Werror -o eval eval.c bv8.c sess.c rand.c giveup.c

gen: gen.c bv8.c giveup.c
	gcc --std=c11 -march=native -Ofast -Werror -o gen gen.c bv8.c giveup.c

gen2: gen2.c bv8.c giveup.c
	gcc --std=c11 -march=native -Ofast -Werror -o gen2 gen2.c bv8.c giveup.c

benchbv8: benchbv8.c bv8.c giveup.c
	gcc --std=c11 -march=native -Ofast -Werror -o benchbv8 benchbv8.c bv8.c giveup.c
