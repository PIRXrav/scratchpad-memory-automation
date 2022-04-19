from optimizer import toeplitz, export
from coalescing_optimizer import run
from prog_gencode import CMvTabGencodeProgVisitor
import numpy as np

exit
DTYPE = "int8_t"
DMA = 128  # 1024o
# Input
x = 16
y = 16

# Filter shape
Dkx = 2
Dky = 2

tensor_i = np.arange(x * y, dtype=np.int32).reshape(y, x)  # tab[index] = index !!
tensor_o = toeplitz(tensor_i, y, x, Dky, Dkx)

# cost, states = run(tensor_i, tensor_o, DMA)
states = [
    [0, 0],
    [0, 256],
    [0, 128],
    [128, 840],
    [128, 768],
    [96, 640],
    [32, 384],
    [64, 512],
]

prog = export(states, tensor_i, tensor_o, DMA)
prog.evaluate(DMA)

citab, cotab, coreloop = CMvTabGencodeProgVisitor(
    prog, "tab0", "tab1", DMA, DTYPE
).export()
print(coreloop)

import subprocess


with open("res.h", "w") as f:
    f.write("/* Generated header */\n")
    f.write("/* Toeplitz matrix transformation test */\n")
    f.write("#pragma once\n")
    f.write('#include "dma.h"\n')
    f.write("\n")
    f.write(f"#define X {x}\n")
    f.write(f"#define Y {y}\n")
    f.write("\n")
    f.write(f"#define TX {tensor_o.shape[1]}\n")
    f.write(f"#define TY {tensor_o.shape[0]}\n")
    f.write(f"#define DKY {Dky}\n")
    f.write(f"#define DKX {Dkx}\n")
    f.write("\n")
    f.write(
        "void toeplitz_naif(__SMA_RAM_PTR void *tab0raw, __SMA_RAM_PTR void *tab1raw){\n"
    )
    f.write(f"    __SMA_RAM_PTR {DTYPE} (*tab0)/*Y*/[X] = tab0raw;\n")
    f.write(f"    __SMA_RAM_PTR {DTYPE} (*tab1)/*TY*/[TX] = tab1raw;\n")
    f.write("\n")
    f.write("    int i_f = 0;\n")
    f.write("    for(int ffy = 0; ffy < (Y - DKY / 2); ffy++){\n")
    f.write("        for(int ffx = 0; ffx < (X - DKX / 2); ffx++){\n")
    f.write("            for(int dkyy = 0; dkyy < DKY; dkyy++){\n")
    f.write("                for(int dkxx = 0; dkxx < DKX; dkxx++){\n")
    f.write(
        "                    tab1[i_f][dkyy * DKX + dkxx] = tab0[ffy + dkyy][ffx + dkxx];\n"
    )
    f.write("                }\n")
    f.write("            }\n")
    f.write("            i_f++;\n")
    f.write("        }\n")
    f.write("    }\n")
    f.write("}\n")
    f.write("\n")
    f.write(citab)
    f.write(cotab)
    f.write(
        "void toeplitz_optim_formv(__SMA_RAM_PTR void *tab0raw, __SMA_RAM_PTR void *tab1raw){\n"
    )
    f.write(f"    __SMA_RAM_PTR {DTYPE} (*tab0)/*Y*/[X] = tab0raw;\n")
    f.write(f"    __SMA_RAM_PTR {DTYPE} (*tab1)/*TY*/[TX] = tab1raw;\n")
    f.write("\n")
    f.write(coreloop)
    f.write("}\n")
    f.write("\n")

subprocess.run(["clang-format", "-i", "res.h"])


with open("res.c", "w") as f:
    f.write("/* Generated tb */\n")
    f.write("/* Toeplitz matrix transformation test */\n")
    f.write("#include <stdio.h>\n")
    f.write("#include <stdlib.h>\n")
    f.write('#include "res.h"\n')
    f.write("\n")
    f.write(f"__SMA_RAM {DTYPE} tab0[Y][X];\n")
    f.write(f"__SMA_RAM {DTYPE} tab1ref[TY][TX]; /* Toeplitz std */")
    f.write(f"__SMA_RAM {DTYPE} tab1[TY][TX]; /* Toeplitz fast */")
    f.write("\n")
    f.write("\n")
    f.write("int64_t postbench(void){\n")
    f.write("    int64_t hash_test = 0;\n")
    f.write("    int64_t hash = 0;\n")
    f.write("    int64_t nb_err = 0;\n")
    f.write("    for (int i = 0; i < TY *TX; i++){")
    f.write(f"       hash += (({DTYPE}*)tab1ref)[i];")
    f.write(f"       hash_test += (({DTYPE}*)tab1)[i];")
    f.write(f"       nb_err += (({DTYPE}*)tab1)[i] != (({DTYPE}*)tab1ref)[i];")
    f.write("    }\n")
    f.write(
        '    printf("PYTHON_RES = {\\"hash\\": %ld, \\"hasht\\": %ld, \\"err\\": %ld}\\n", hash, hash_test, nb_err);\n'
    )
    f.write("    return nb_err ? 42 : 0;\n")
    f.write("}\n")
    f.write("void prebench(void){\n")
    f.write("    for (int i = 0; i < Y * X; i++){")
    f.write(f"        (({DTYPE}*)tab0)[i] = ({DTYPE})i;")
    f.write("    }")
    f.write('    printf("tab0    = %p # %ld\\n", tab0, sizeof(tab0));')
    f.write('    printf("tab1ref = %p # %ld\\n", tab1ref, sizeof(tab1ref));')
    f.write('    printf("tab1    = %p # %ld\\n", tab1, sizeof(tab1));')
    f.write("}\n")
    f.write("int main(void){{\n")
    f.write('    printf("Run benchmark\\n");\n')
    f.write("    /* prebench */\n")
    f.write('    printf("prebench ...\\n");\n')
    f.write("    prebench();")
    f.write("    /* Run */\n")
    f.write('    printf("Run ...\\n");\n')
    f.write("    toeplitz_optim_formv(tab0, tab1);\n")
    f.write("    toeplitz_naif(tab0, tab1ref);\n")
    f.write("    /* postbench */\n")
    f.write('    printf("postbench ...\\n");\n')
    f.write("    exit(postbench());\n")
    f.write("}}\n")

subprocess.run(["clang-format", "-i", "res.c"])

from toolchain import shell

PREFIX = "/tmp/"
SMA_SOURCE = PREFIX + "sma_source.c"
SMA_BIN = PREFIX + "sma_bin"

CC = "gcc"
CFLAGS = "-Wall -Wextra -Werror -Idmasimulator -g -O1"
LDFLAGS = ""

cmd = f"{CC} {CFLAGS} {LDFLAGS} res.c -I. -o {SMA_BIN}"
shell(cmd, verbose=True)
