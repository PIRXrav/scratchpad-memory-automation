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

cprog = CLinearGencodeProgVisitor(prog, 'tab0', 'tab1', '__SMA__dma0', '__SMA__dma1', DMA)
print(cprog)
