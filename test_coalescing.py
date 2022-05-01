from optimizer import toeplitz, export
from coalescing_optimizer import run
from prog_gencode import CMvTabGencodeProgVisitor, GenericGencodeProgVisitor
import numpy as np
import ctools
import toolchain as tc

class Coalescing:
    def __init__(self, tensor_i, tensor_o, dtype, type_size, dma, word_size):
        self.tensor_i = tensor_i
        self.tensor_o = tensor_o
        self.dtype = dtype
        self.type_size = type_size
        self.dma = dma
        self.word_size = word_size

        # Compute state path
        self.cost = 24
        self.states = [
            [0, 0],
            [0, 256],
            [0, 128],
            [128, 840],
            [128, 768],
            [96, 640],
            [32, 384],
            [64, 512],
        ]
        self.cost, self.states = run(tensor_i, tensor_o, self.word_size, self.dma)
        self.prog = export(self.states, self.tensor_i, self.tensor_o, self.dma, self.word_size)

    def export(self):
        return self.prog

    def benchmark(self):
        from toolchain import gcc, write_c_file, write_file, shell
        from ctools import comment_header

        evaluation = self.prog.gen_evaluation(self.dma)
        hist = self.prog.gen_hist_dma_repartition()
        print(self.prog)
        if 1:
            framechain = GenericGencodeProgVisitor(self.prog, self.dma, self.word_size).export()
            base_size, arr = framechain.as_array(self.word_size)
            prog_arr_c = ctools.bstr_to_c('_sma_prog0', arr)
            func = f'prog_mv{self.type_size}'
            print(framechain)
            print(base_size)
            print(arr, len(arr))
            print(prog_arr_c)
            decl_c = prog_arr_c
            prog_c = f'{func}(tensor_i, tensor_o_test, _sma_prog0, {base_size});\n'
            incs_c = '#include "prog_mv.h"'
        else:
            progcode = CMvTabGencodeProgVisitor(self.prog, "tensor_i", "tensor_o_test", self.dma, self.dtype)
            citab, cotab, coreloop = progcode.export()
            decl_c = citab + cotab
            prog_c = coreloop
            incs_c = ''

        code = comment_header("GENERATED FILE: res.h",
                              file="Toeplitz matrix transformation test",
                              **evaluation, rw=hist)
        code += "#pragma once\n"
        code += '#include "dma.h"\n'
        code += incs_c
        code += "\n"
        code += f"#define X {x}\n"
        code += f"#define Y {y}\n"
        code += "\n"
        code += f"#define TX {tensor_o.shape[1]}\n"
        code += f"#define TY {tensor_o.shape[0]}\n"
        code += f"#define DKY {Dky}\n"
        code += f"#define DKX {Dkx}\n"
        code += "\n"
        code += "\n"
        code += decl_c
        code += "void toeplitz_optim_formv(__SMA_RAM_PTR void *tensor_iraw, __SMA_RAM_PTR void *tensor_o_testraw){\n"
        code += f"    __SMA_RAM_PTR {DTYPE} (*tensor_i)/*Y*/[X] = tensor_iraw;\n"
        code += f"    __SMA_RAM_PTR {DTYPE} (*tensor_o_test)/*TY*/[TX] = tensor_o_testraw;\n"
        code += "\n"
        code += prog_c
        code += "}\n"
        code += "\n"
        write_c_file(tc.PATH_GEN_FILES + "res.h", code, display_code=True)

        tensor_o_test = np.zeros(tensor_o.size, dtype=np.int32).reshape(tensor_o.shape)
        code = comment_header("GENERATED FILE: res.c",
                              file="Toeplitz matrix transformation test",
                              **evaluation, rw=hist)
        code += "#include <stdio.h>\n"
        code += "#include <stdlib.h>\n"
        code += '#include "res.h"\n'
        code += "\n"
        code += f"__SMA_RAM {ctools.nparray_to_c(DTYPE, 'tensor_i', tensor_i)}"
        code += f"__SMA_RAM {ctools.nparray_to_c(DTYPE, 'tensor_o', tensor_o)}"
        code += f"__SMA_RAM {ctools.nparray_to_c(DTYPE, 'tensor_o_test', tensor_o_test)}"
        code += "\n"
        code += "\n"
        code += "int64_t postbench(void){\n"
        code += "    int64_t hash_test = 0;\n"
        code += "    int64_t hash = 0;\n"
        code += "    int64_t nb_err = 0;\n"
        code += "    for (int i = 0; i < TY *TX; i++){"
        code += f"       hash += (({DTYPE}*)tensor_o)[i];"
        code += f"       hash_test += (({DTYPE}*)tensor_o_test)[i];"
        code += f"       nb_err += (({DTYPE}*)tensor_o_test)[i] != (({DTYPE}*)tensor_o)[i];"
        code += "    }\n"
        code += '    printf("PYTHON_RES = {\\"hash\\": %ld, \\"hasht\\": %ld, \\"err\\": %ld}\\n", hash, hash_test, nb_err);\n'
        code += "    return nb_err ? 42 : 0;\n"
        code += "}\n"
        code += "void prebench(void){\n"
        code += "    for (int i = 0; i < Y * X; i++){"
        code += f"        (({DTYPE}*)tensor_i)[i] = ({DTYPE})i;"
        code += "    }"
        code += '    printf("tensor_i    = %p # %ld\\n", tensor_i, sizeof(tensor_i));'
        code += '    printf("tensor_o = %p # %ld\\n", tensor_o, sizeof(tensor_o));'
        code += '    printf("tensor_o_test    = %p # %ld\\n", tensor_o_test, sizeof(tensor_o_test));'
        code += "}\n"
        code += "int main(void){{\n"
        code += '    printf("Run benchmark\\n");\n'
        code += "    /* prebench */\n"
        code += '    printf("prebench ...\\n");\n'
        code += "    prebench();"
        code += "    /* Run */\n"
        code += '    printf("Run ...\\n");\n'
        code += "    toeplitz_optim_formv(tensor_i, tensor_o_test);\n"
        code += "    /* postbench */\n"
        code += '    printf("postbench ...\\n");\n'
        code += "    exit(postbench());\n"
        code += "}}\n"
        tc.write_c_file(tc.PATH_GEN_FILES + "res.c", code, display_code=False)

        gdbname, code, dump_file_names = self.generate_benchmark_gdb()
        tc.write_file(tc.PATH_GEN_FILES + gdbname, code)
        # build
        # gcc(['res.c'], binfile, opts='-DHW_WORD_CONSTRAINTS', verbose=True)
        tc.make(src='genfiles/res.c')

        if 1:
            args = f'USER_GDB={"genfiles/" + gdbname} gdb'
        else:
            args = 'run'
        res, _ = tc.python_res_catch(tc.make(src='genfiles/res.c', args=args))
        print(res)
        print("SUCCESS" if res['err'] == 0 else "ERROR")
        return int(res['err'])

    def generate_benchmark_gdb(self, prefix=""):
        decl_names = ('tensor_i', 'tensor_o', 'tensor_o_test')
        dump_file_names = [prefix + d + ".bin" for d in decl_names]
        print(dump_file_names)

        filename = 'koala' + ".gdb"
        code = ""
        code += "b postbench\n"
        code += "run\n"
        for dn, dump_file in zip(decl_names, dump_file_names):
            code += f"dump binary memory {dump_file} {dn} (char*){dn} + sizeof({dn})\n"
        code += "c\n"
        return filename, code, dump_file_names

# DMA config
WORD_SIZE = 8
DMA = 128  # 1024o

DTYPE = "int8_t"
DTYPE_SIZE = 8


# Input
x = 16
y = 16

# Filter shape
Dkx = 1
Dky = 1

np.random.seed(12)

tensor_i = np.arange(x * y, dtype=np.int32).reshape(y, x)  # tab[index] = index !!
tensor_o = toeplitz(tensor_i, y, x, Dky, Dkx)
# tensor_o = np.random.randint(tensor_i.size, size=(4, 4))  # Random shape with tensor_i values
# tensor_o = tensor_i.copy()
# tensor_o = np.zeros((3, 3))
Coala = Coalescing(tensor_i, tensor_o, DTYPE, DTYPE_SIZE, DMA, WORD_SIZE)
Coala.benchmark()
