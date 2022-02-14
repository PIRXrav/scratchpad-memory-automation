from optimizer import algo1, toeplitz, export, DMA
from gencode import CLinearGencodeProgVisitor
import numpy as np

# Input
x = 8
y = 8

# Filter shape
Dkx = 2
Dky = 2

tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
tensor_o = toeplitz(tensor_i, Dky, Dkx)

best = algo1(y, x, Dky, Dkx, v_fast_explore_numba=True)
prog = export(best[1], tensor_o)
prog.evaluate(DMA)

# TODO remove cast & DMA * 4
cprog = CLinearGencodeProgVisitor(prog, '(int32_t*)tab0', '(int32_t*)tab1', '(int32_t*)__SMA__dma0', '(int32_t*)__SMA__dma1', DMA*4)
print(cprog)




with open('res.c', 'w') as f:
    f.write('/* Generated f */\n')
    f.write('/* Toeplitz matrix transformation test */\n')
    f.write('#include "dma.h"\n')
    f.write('#include <stdio.h>\n')
    f.write('\n')
    f.write(f'#define X {x}\n')
    f.write(f'#define Y {y}\n')   
    f.write('\n')
    f.write(f'#define TX {tensor_o.shape[1]}\n')
    f.write(f'#define TY {tensor_o.shape[0]}\n')
    f.write(f'#define DKY {Dky}\n')
    f.write(f'#define DKX {Dkx}\n')
    f.write('\n')  
    f.write('int32_t tab0[Y][X];\n')  
    f.write('int32_t tab1ref[TY][TX]; /* Toeplitz std */\n')
    f.write('int32_t tab1[TY][TX]; /* Toeplitz fast */\n') 
    f.write('\n')  
    f.write('\n')
    f.write('int main(void){\n')
    # Init tab0
    f.write('    for(int i = 0; i < Y*X; i++){((int32_t*)tab0)[i] = i;}')
    # Perform ref translation
    f.write('    int i_f = 0;\n')
    f.write('    for(int ffy = 0; ffy < (Y - DKY / 2); ffy++){\n')
    f.write('        for(int ffx = 0; ffx < (X - DKX / 2); ffx++){\n')
    f.write('            for(int dkyy = 0; dkyy < DKY; dkyy++){\n')
    f.write('                for(int dkxx = 0; dkxx < DKX; dkxx++){\n')
    f.write('                    tab1ref[i_f][dkyy * DKX + dkxx] = tab0[ffy + dkyy][ffx + dkxx];\n')
    f.write('                }\n')
    f.write('            }\n')
    f.write('            i_f++;\n')
    f.write('        }\n')
    f.write('    }\n')
    f.write('\n')
    f.write(cprog.export())
    f.write('\n')
    f.write('for(int i = 0; i < Y*X; i++){printf("tab1ref=%d\\ttab1=%d\\n", ((int32_t*)tab1ref)[i], ((int32_t*)tab1)[i]);}')
    f.write('    return 0;\n')
    f.write('}\n')




            
          

    
    
