from optimizer import algo1, toeplitz, export
from gencode import CLinearGencodeProgVisitor, CMvTabGencodeProgVisitor
import numpy as np

exit

DMA = 128 # 1024o
# Input
x = 16
y = 16

# Filter shape
Dkx = 2
Dky = 2

tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
tensor_o = toeplitz(tensor_i,  y, x, Dky, Dkx)

best = algo1(y, x, Dky, Dkx, DMA, v_fast_explore_numba=True)
prog = export(best[1], tensor_i, tensor_o, DMA)
prog.evaluate(DMA)

# TODO remove cast & DMA * 4
cprog = CLinearGencodeProgVisitor(prog, 'tab0', 'tab1', 'local_dmai', 'local_dmao', DMA * 8)
# print(cprog)


citab, cotab, coreloop = CMvTabGencodeProgVisitor(prog, 'tab0', 'tab1', 'local_dmai', 'local_dmao', DMA * 8).export()
print(coreloop)

import subprocess



with open('res.h', 'w') as f:
    f.write('/* Generated header */\n')
    f.write('/* Toeplitz matrix transformation test */\n')
    f.write('#pragma once\n')
    f.write('#include "dma.h"\n')
    f.write('\n')
    f.write(f'#define X {x}\n')
    f.write(f'#define Y {y}\n')   
    f.write('\n')
    f.write(f'#define TX {tensor_o.shape[1]}\n')
    f.write(f'#define TY {tensor_o.shape[0]}\n')
    f.write(f'#define DKY {Dky}\n')
    f.write(f'#define DKX {Dkx}\n')
    f.write('\n')  
    f.write('void toeplitz_naif(__SMA__ram_ptr void *tab0raw, __SMA__ram_ptr void *tab1raw){\n')
    f.write('    __SMA__ram_ptr const int64_t (*tab0)/*Y*/[X] = tab0raw;\n')
    f.write('    __SMA__ram_ptr int64_t (*tab1)/*TY*/[TX] = tab1raw;\n')
    f.write('\n')
    f.write('    int i_f = 0;\n')
    f.write('    for(int ffy = 0; ffy < (Y - DKY / 2); ffy++){\n')
    f.write('        for(int ffx = 0; ffx < (X - DKX / 2); ffx++){\n')
    f.write('            for(int dkyy = 0; dkyy < DKY; dkyy++){\n')
    f.write('                for(int dkxx = 0; dkxx < DKX; dkxx++){\n')
    f.write('                    tab1[i_f][dkyy * DKX + dkxx] = tab0[ffy + dkyy][ffx + dkxx];\n')
    f.write('                }\n')
    f.write('            }\n')
    f.write('            i_f++;\n')
    f.write('        }\n')
    f.write('    }\n')
    f.write('}\n')
    f.write('\n')
    f.write(citab)
    f.write(cotab)
    f.write('void toeplitz_optim_formv(__SMA__ram_ptr void *tab0raw, __SMA__ram_ptr void *tab1raw, int64_t *local_dmao, int64_t *local_dmai){\n')
    f.write('    __SMA__ram_ptr const int64_t (*tab0)/*Y*/[X] = tab0raw;\n')
    f.write('    __SMA__ram_ptr int64_t (*tab1)/*TY*/[TX] = tab1raw;\n')
    f.write('\n')
    f.write(coreloop)
    f.write('}\n')
    f.write('\n')

subprocess.run(["clang-format", '-i', 'res.h'])
    
