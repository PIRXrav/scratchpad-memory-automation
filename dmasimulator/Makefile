###
# Generic makefile
###


SRC 	= $(USER_SRC)
INC 	= -Isimdma/include -I. $(USER_INC)
OBJ 	= $(addprefix build/,$(SRC:.c=.o)) ./simdma/lib/libsimdma.a
CFLAGS 	+= -Wall -Wextra -Werror -g $(INC)
LDFLAGS = $(CFLAGS)
CC 		= gcc
LD 		= gcc
EXEC 	= build/app


all: $(EXEC)

$(EXEC): $(OBJ)
	mkdir -p $(@D)
	$(LD) $(LDFLAGS) -o $@ $^

./simdma/lib/libsimdma.a: libs

build/%.o: %.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

.phony: libs
libs:
	make -C simdma

# -------------------------------- run --------------------------------------- #

.phony: run
run: $(EXEC)
	./$<


# -------------------------------- gdb --------------------------------------- #

.phony: gdb
gdb: $(EXEC) $(USER_GDB)
	cd build && gdb --batch --command=../$(USER_GDB) gdbname --args app

$(USER_GDB):
	$(error missing GDB script: $@)


clean:
	rm -rf build
	make -C simdma clean