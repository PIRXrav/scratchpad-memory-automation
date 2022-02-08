
# Analyse memory transaction during toeplitz transformation

import numpy as np


# Input
x = 3
y = 3
tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
print(tensor_i)

# Filter shape
Dkx = 2
Dky = 2

# Toeplitz matrix
o_x = Dkx * Dky
o_y = (x - Dkx // 2) * (y - Dky // 2)
tensor_o = np.zeros((o_y, o_x), dtype=np.int32)
print(tensor_o)

# Perform translation
for ffy in range(y - Dky // 2):
    for ffx in range(x - Dkx // 2):
        i_filter = ffy * (x - Dkx // 2) + ffx
        for dkyy in range(Dky):
            for dkxx in range(Dkx):
                tensor_o[i_filter][dkyy * Dkx + dkxx] = tensor_i[ffy + dkyy][ffx + dkxx]
print(tensor_o)
print(tensor_o.shape)

# use 5 word DMA buffer
DMA = 5
def dma_load(tensor, addr):
    return tensor.ravel()[addr:min(addr+DMA,tensor.size)]

def dma_store(tensor, addr, dma):
    tensor.ravel()[addr:min(addr+DMA,tensor.size)] = dma

print(f'{dma_load(tensor_i, 0)=}')
print(f'{dma_load(tensor_i, 4)=}')
print(f'{dma_load(tensor_i, 15)=}')

tensor_i_dma = dma_load(tensor_i, 0)
tensor_o_dma = dma_load(tensor_o, 8)
print(tensor_i_dma, tensor_o_dma)

def compare_dma(dma_i, dma_o):
    return np.intersect1d(dma_i, dma_o)

def dma_catch(dma, values):
    # Beark !
    for i in range(len(dma)):
        if dma[i] in values:
            dma[i] = -1
    # tensor.ravel()[linear_indexs] = -1 #

def is_tensor_fully_catched(tensor):
    return np.all(tensor == -1)


dma_match = compare_dma(tensor_i_dma, tensor_o_dma)
print(f'{dma_match=}')

print(tensor_o)
dma_catch(tensor_o_dma, dma_match)
print(tensor_i_dma, tensor_o_dma)

is_tensor_fully_catched(tensor_o)
print(tensor_o)



from itertools import product

# Compute all combinations of @input, @output (we will use remove on it -> set)
combination_dma_io = set(product(range(tensor_i.size), range(tensor_o.size)))

combination_dma_io_valid = set()

for comb in combination_dma_io:
    tensor_i_dma = dma_load(tensor_i, comb[0])
    tensor_o_dma = dma_load(tensor_o, comb[1])
    diff = compare_dma(tensor_i_dma, tensor_o_dma)
    if diff.size:
        print(f'{comb}: {tensor_i_dma} && {tensor_o_dma} = {diff})')
        combination_dma_io_valid.add(comb)

from numba import jit
# @jit(nopython=True)

def compute_available_next(cur: tuple[int, int], combs: set) -> list:
    if cur == (-1, -1):
        return combs
    return set(filter(lambda x: x[0] == cur[0] or x[1] == cur[1], combs))


print(f'{combination_dma_io=}')

# Tous les états du système ou les DMA matchs !
print(f'{combination_dma_io_valid=}')
print(f'{compute_available_next((0, 0), combination_dma_io_valid)=}')

max_q = 0


def compute(current_state, dma_i, dma_o, system_states, tensor_o, cost, dept, state_history):
    s = '.' * dept + f'{np.sum(tensor_o==-1)}' + str(state_history)

    # print(f'{s:<80} hh')
    # print(dept, np.sum(tensor_o==-1), '/', tensor_o.size, tensor_o.ravel())

    res = []
    if dept > 16+1:
        return 
    # input()
    next_states = set(compute_available_next(current_state, system_states))
    for i, state in enumerate(next_states):
        # Simulate the new state:
        op_cost = 0
        if current_state[0] != state[0]: # Load input dma
            op_cost += 1
        if current_state[1] != state[1]: # Load and store output dma
            op_cost += 2
        
        
        dma_i = dma_load(tensor_i, state[0])
        dma_o = dma_load(tensor_o, state[1])
        dma_match = compare_dma(dma_i, dma_o)
        # print(state, dma_i, dma_o, dma_match)
        if dma_match.size: # This state has interest
            next_tensor_o = np.copy(tensor_o) # We need to preserve the state
            dma_o = dma_load(next_tensor_o, state[1]) # Get the new reference
            dma_catch(dma_o, dma_match) # Catch

            state_history.append(state) # Lifo push
            if is_tensor_fully_catched(next_tensor_o):
                print('HIT!!!', state_history, cost, dept)
                
                # return current_state

            ress = compute(state, dma_i, dma_o, system_states - {state}, next_tensor_o, cost + op_cost, dept+1, state_history)
            state_history.pop() # Lifo pop
            res.append(ress)
        else:
            pass
    return res

if 1:
    compute((-1, -1),                 # Do not force initial state
            None,                     # No DMA provided
            None,                     #
            combination_dma_io_valid, # All state te explore
            tensor_o,                 # Default result
            0, 0, [])

        


