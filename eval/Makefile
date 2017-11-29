all: eval gen gen2 benchbv8

eval: eval.c bv8.c sess.c rand.c giveup.c
	gcc --std=c11 -march=native -Ofast -Werror -o eval eval.c bv8.c sess.c rand.c giveup.c

gen: gen.c bv8.c giveup.c hash.c
	gcc --std=c11 -march=native -Ofast -pg -Werror -o gen gen.c bv8.c giveup.c hash.c

gen2: gen2.c bv8.c giveup.c hash.c
	gcc --std=c11 -march=native -Ofast -pg -Werror -o gen2 gen2.c bv8.c giveup.c hash.c

benchbv8: benchbv8.c bv8.c giveup.c
	gcc --std=c11 -march=native -mmmx -msse -msse2 -msse3 -Ofast -Werror -o benchbv8 benchbv8.c bv8.c giveup.c

fast: eval-fast gen-fast gen2-fast benchbv8-fast

eval-fast: eval.c bv8-avx2.c sess.c rand.c giveup.c
	gcc --std=c11 -march=native -Ofast -Werror -o eval-fast eval.c bv8-avx2.c sess.c rand.c giveup.c

gen-fast: gen.c bv8-avx2.c giveup.c hash-rbtree.c
	gcc --std=c11 -march=native -Ofast -pg -Werror -o gen-fast gen.c bv8-avx2.c giveup.c hash-rbtree.c

gen2-fast: gen2.c bv8-avx2.c giveup.c hash-rbtree.c
	gcc --std=c11 -march=native -Ofast -pg -Werror -o gen2-fast gen2.c bv8-avx2.c giveup.c hash-rbtree.c

benchbv8-fast: benchbv8.c bv8-avx2.c giveup.c
	gcc --std=c11 -march=native -mmmx -msse -msse2 -msse3 -Ofast -Werror -o benchbv8-fast benchbv8.c bv8-avx2.c giveup.c
