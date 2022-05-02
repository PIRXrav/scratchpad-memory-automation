###
# Generic makefile
#
# RULES
#		all
#			Build main application
#
#		run
#			Run main application
#
#		gdb
#			Run gdb scipt over main application
#
# MANDATORY ARGS
#		USER_DMA_SIZE
#			Dma syze in bytes
#
#		USER_WORD_SIZE
#			Dma word size: memory LD/ST addr must be multiple this value
#
# ARGS
#		USER_SRC
#			List of user c files
#
#		USER_INC
#			List of -I<folder>
#
#		USER_GDB
#			path of a gdb script
#
###

# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

ifneq ($(firstword $(MAKECMDGOALS)),clean)
$(call check_defined, USER_DMA_SIZE, USER_DMA_SIZE must be defined)
$(call check_defined, USER_WORD_SIZE, USER_WORD_SIZE must be defined)
endif

CC 				:= gcc
LD 				:= gcc
CFLAGS			:= -Wall -Wextra -Werror -g -O1 \
					-DDMA_SIZE=$(USER_DMA_SIZE) -DWORD_SIZE=$(USER_WORD_SIZE)
LDFLAGS			:=
BUILD			:= build


SIMDMA_KEY 		:= $(USER_DMA_SIZE)_$(USER_WORD_SIZE)

SIMDMA_SRC 		:= $(shell find simdma/src/ -type f -name '*.c')
SIMDMA_OBJ 		:= $(patsubst simdma/src/%.c,$(BUILD)/simdma/%_$(SIMDMA_KEY).o,$(SIMDMA_SRC))
SIMDMA_DEP 		:= $(SIMDMA_OBJ:%.o=%.d)
SIMDMA_LIB 		:= $(BUILD)/libsimdma_$(SIMDMA_KEY).a
SIMDMA_CFLAGS 	:= $(CFLAGS) -Isimdma/src/ -Isimdma/include/ -fstack-protector


APP_SRC 		:= $(USER_SRC)
APP_INC 		:= -Isimdma/include -I. $(USER_INC)
APP_OBJ 		:= $(patsubst %.c,$(BUILD)/%.o,$(APP_SRC)) $(SIMDMA_LIB)
APP_DEP			:= $($(filter .o, $(APP_OBJ)):%.o=%.d)
APP_CFLAGS 		:= $(CFLAGS) $(APP_INC)
APP_LDFLAGS 	:= $(APP_CFLAGS)
EXEC 			:= $(BUILD)/app

# default rule
all: $(EXEC)
	@echo $^ finshed

# ##############################################################################
# dma simulator
# ##############################################################################

# Dependancies
-include $(SIMDMA_DEP)

$(BUILD)/simdma/%_$(SIMDMA_KEY).o: simdma/src/%.c
	mkdir -p $(@D)
	$(CC) $(SIMDMA_CFLAGS) $(EXTRA_CFLAGS) -MMD -c $< -o $@ 

$(SIMDMA_LIB): $(SIMDMA_OBJ)
	mkdir -p $(@D)
	ar -rv $@ $^


# ##############################################################################
# User application
# ##############################################################################

# Dependancies
-include $(APP_DEP)

# ------------------------------- build -------------------------------------- #

$(EXEC): $(APP_OBJ)
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) -o $@ $^

build/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(APP_CFLAGS) -MMD -c $< -o $@

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


# ##############################################################################
# Misc
# ##############################################################################


.PHONY: clean
clean:
	rm -rf build
