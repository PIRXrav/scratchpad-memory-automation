from optimizer import toeplitz, export
from coalescing_optimizer import run
from prog_gencode import CMvTabGencodeProgVisitor
import numpy as np



# DMA config
WORD_SIZE = 8
DMA = 128  # 1024o


# Input
x = 16
y = 16

# Filter shape
Dkx = 2
Dky = 2

DTYPE = "int8_t"

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

prog = export(states, tensor_i, tensor_o, DMA, WORD_SIZE)
evaluation = prog.gen_evaluation(DMA)
hist = prog.gen_hist_dma_repartition()

progcode = CMvTabGencodeProgVisitor(prog, "tab0", "tab1", DMA, DTYPE)
citab, cotab, coreloop = progcode.export()
print(coreloop)

from toolchain import gcc, write_c_file, benchmarkrun
from ctools import comment_header

code = comment_header("GENERATED FILE: res.h",
                      file="Toeplitz matrix transformation test",
                      **evaluation, rw=hist)
code += "#pragma once\n"
code += '#include "dma.h"\n'
code += "\n"
code += f"#define X {x}\n"
code += f"#define Y {y}\n"
code += "\n"
code += f"#define TX {tensor_o.shape[1]}\n"
code += f"#define TY {tensor_o.shape[0]}\n"
code += f"#define DKY {Dky}\n"
code += f"#define DKX {Dkx}\n"
code += "\n"
code += "void toeplitz_naif(__SMA_RAM_PTR void *tab0raw, __SMA_RAM_PTR void *tab1raw){\n"
code += f"    __SMA_RAM_PTR {DTYPE} (*tab0)/*Y*/[X] = tab0raw;\n"
code += f"    __SMA_RAM_PTR {DTYPE} (*tab1)/*TY*/[TX] = tab1raw;\n"
code += "\n"
code += "    int i_f = 0;\n"
code += "    for(int ffy = 0; ffy < (Y - DKY / 2); ffy++){\n"
code += "        for(int ffx = 0; ffx < (X - DKX / 2); ffx++){\n"
code += "            for(int dkyy = 0; dkyy < DKY; dkyy++){\n"
code += "                for(int dkxx = 0; dkxx < DKX; dkxx++){\n"
code += "                    tab1[i_f][dkyy * DKX + dkxx] = tab0[ffy + dkyy][ffx + dkxx];\n"
code += "                }\n"
code += "            }\n"
code += "            i_f++;\n"
code += "        }\n"
code += "    }\n"
code += "}\n"
code += "\n"
code += citab
code += cotab
code += "void toeplitz_optim_formv(__SMA_RAM_PTR void *tab0raw, __SMA_RAM_PTR void *tab1raw){\n"
code += f"    __SMA_RAM_PTR {DTYPE} (*tab0)/*Y*/[X] = tab0raw;\n"
code += f"    __SMA_RAM_PTR {DTYPE} (*tab1)/*TY*/[TX] = tab1raw;\n"
code += "\n"
code += coreloop
code += "}\n"
code += "\n"
write_c_file("res.h", code, display_code=True)

code = comment_header("GENERATED FILE: res.c",
                      file="Toeplitz matrix transformation test",
                      **evaluation, rw=hist)
code += "#include <stdio.h>\n"
code += "#include <stdlib.h>\n"
code += '#include "res.h"\n'
code += "\n"
code += f"__SMA_RAM {DTYPE} tab0[Y][X];\n"
code += f"__SMA_RAM {DTYPE} tab1ref[TY][TX]; /* Toeplitz std */"
code += f"__SMA_RAM {DTYPE} tab1[TY][TX]; /* Toeplitz fast */"
code += "\n"
code += "\n"
code += "int64_t postbench(void){\n"
code += "    int64_t hash_test = 0;\n"
code += "    int64_t hash = 0;\n"
code += "    int64_t nb_err = 0;\n"
code += "    for (int i = 0; i < TY *TX; i++){"
code += f"       hash += (({DTYPE}*)tab1ref)[i];"
code += f"       hash_test += (({DTYPE}*)tab1)[i];"
code += f"       nb_err += (({DTYPE}*)tab1)[i] != (({DTYPE}*)tab1ref)[i];"
code += "    }\n"
code += '    printf("PYTHON_RES = {\\"hash\\": %ld, \\"hasht\\": %ld, \\"err\\": %ld}\\n", hash, hash_test, nb_err);\n'
code += "    return nb_err ? 42 : 0;\n"
code += "}\n"
code += "void prebench(void){\n"
code += "    for (int i = 0; i < Y * X; i++){"
code += f"        (({DTYPE}*)tab0)[i] = ({DTYPE})i;"
code += "    }"
code += '    printf("tab0    = %p # %ld\\n", tab0, sizeof(tab0));'
code += '    printf("tab1ref = %p # %ld\\n", tab1ref, sizeof(tab1ref));'
code += '    printf("tab1    = %p # %ld\\n", tab1, sizeof(tab1));'
code += "}\n"
code += "int main(void){{\n"
code += '    printf("Run benchmark\\n");\n'
code += "    /* prebench */\n"
code += '    printf("prebench ...\\n");\n'
code += "    prebench();"
code += "    /* Run */\n"
code += '    printf("Run ...\\n");\n'
code += "    toeplitz_optim_formv(tab0, tab1);\n"
code += "    toeplitz_naif(tab0, tab1ref);\n"
code += "    /* postbench */\n"
code += '    printf("postbench ...\\n");\n'
code += "    exit(postbench());\n"
code += "}}\n"
write_c_file("res.c", code, display_code=False)


binfile = '/tmp/coalescingtb'
gcc(['res.c'], binfile, opts='-DHW_WORD_CONSTRAINTS', verbose=True)
res, out = benchmarkrun(binfile, verbose=False)
print("SUCCESS" if res['err'] == 0 else "ERROR")
print(res)
print(out)
