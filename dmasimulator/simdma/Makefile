# DPU sources
SRC = $(shell find src/ -type f -name '*.c')
OBJ = $(patsubst src/%.c,build/%.o,$(SRC))
DEP = $(OBJ:%.o=%.d)
LIB = lib/libsimdma.a

CC = gcc
LD = gcc

CFLAGS =  -g -Wall -Wextra -Werror -Isrc/ -Iinclude/ -fstack-protector
LDFLAGS = 

all: $(LIB)

# Dependancies
-include $(DEP)

# The -MMD flags additionaly creates a .d file with same name
build/%.o: src/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -MMD -c $< -o $@ 

$(LIB): $(OBJ)
	mkdir -p $(@D)
	ar -rv $@ $^

clean:
	rm -rf $(LIB)
	rm -rf build

.PHONY: clean
