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
app 	= build/app


all: $(app)

$(app): $(OBJ)
	mkdir -p $(@D)
	$(LD) $(LDFLAGS) -o $@ $^

./simdma/lib/libsimdma.a: libs

build/%.o: %.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

.phony: libs
libs:
	make -C simdma

.phony: run
run: $(app)
	./$<

clean:
	rm -rf build
	make -C simdma clean