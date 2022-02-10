
# Analyse memory transaction during toeplitz transformation

import numpy as np

import sys
import enlighten
from itertools import product


# Input
x = 3
y = 3
tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
print(tensor_i)

# Filter shape
Dkx = 2
Dky = 2

    
# use 5 word DMA buffer
DMA = 5

def toeplitz(tensor_i, Dky, Dkx):
    # Toeplitz matrix
    o_x = Dkx * Dky
    o_y = (x - Dkx // 2) * (y - Dky // 2)
    tensor_o = np.zeros((o_y, o_x), dtype=np.int32)

    # Perform translation
    for ffy in range(y - Dky // 2):
        for ffx in range(x - Dkx // 2):
            i_filter = ffy * (x - Dkx // 2) + ffx
            for dkyy in range(Dky):
                for dkxx in range(Dkx):
                    tensor_o[i_filter][dkyy * Dkx + dkxx] = tensor_i[ffy + dkyy][ffx + dkxx]

    return tensor_o

def dma_load(tensor, addr):
    return tensor.ravel()[addr:min(addr+DMA,tensor.size)]

def dma_store(tensor, addr, dma):
    tensor.ravel()[addr:min(addr+DMA,tensor.size)] = dma

def compare_dma(dma_i, dma_o):
    return np.intersect1d(dma_i, dma_o)

def dma_catch(dma, values):
    # Beark !
    for i in range(len(dma)):
        if dma[i] in values:
            dma[i] = -1

def is_tensor_fully_catched(tensor):
    return np.all(tensor == -1)

tensor_o = toeplitz(tensor_i, Dky, Dkx)
print(tensor_o)
print(tensor_o.shape)

# print(f'{dma_load(tensor_i, 0)=}')
# print(f'{dma_load(tensor_i, 4)=}')
# print(f'{dma_load(tensor_i, 15)=}')

# tensor_i_dma = dma_load(tensor_i, 0)
# tensor_o_dma = dma_load(tensor_o, 8)
# print(tensor_i_dma, tensor_o_dma)


# dma_match = compare_dma(tensor_i_dma, tensor_o_dma)
# print(f'{dma_match=}')

# print(tensor_o)
# dma_catch(tensor_o_dma, dma_match)
# print(tensor_i_dma, tensor_o_dma)

# is_tensor_fully_catched(tensor_o)
# print(tensor_o)

from numba import jit


def progressbar_init(manager, names):
    return [manager.counter(total=100, desc=n, unit=n, color="green") for n in names]
    
def progressbar(iterator, dept, tk):
    if len(tk) > dept:
        tk[dept].total = len(iterator)
        tk[dept].count = 0
        
    for v in iterator:
        if len(tk) > dept:
            tk[dept].update()
        yield v

def export_algo(states, tensor_o):
    state = [-1, -1]
    prog = []

    def transactions(tensor_o):
        dmai = dma_load(tensor_i, state[0])
        dmao = dma_load(tensor_o, state[1])
        for idmai, vi in enumerate(dmai):
            for idmao, vo in enumerate(dmao):
                if vi == vo:
                    dmao[idmao] = -1 # In reality, copy value:
                    prog.append(('mv', idmai, idmao)) #  += f'dma_o[{idmao}] = dma_i[{idmai}]\n' 
        return prog

    for i, o in states:
        if state[0] != i:
            prog.append(('ldi', i))
            # prog += f'DMA_LD(dma_i, {i})\n'
            state[0] = i
            transactions(tensor_o)

        if state[1] != o:
            if state[1] != -1: # Small optim
                prog.append(('sto', state[1]))
                # prog += f'DMA_ST(dma_o, {state[1]})\n'
            prog.append(('ldo', o))
            # prog += f'DMA_LD(dma_o, {o})\n'
            state[1] = o
            transactions(tensor_o)
    
    prog.append(('sto', o))
    assert np.all(tensor_o == -1)
    return prog

def algo0(y, x, Dky, Dkx):
    
    def compute(current_state, system_states, tensor_o, cost, dept, history_stack, best, state_eval_all):

        # Heuristique
        def compute_available_next(cur: tuple[int, int], combs: set) -> list:
            if cur == (-1, -1):
                return combs
            return set(filter(lambda x: x[0] == cur[0] or x[1] == cur[1], combs))

        if np.all(tensor_o == -1): # Done
            if cost < best[0]:
                best[0] = cost
                best[1] = history_stack[:dept]
                print(f'HIT [{cost}/{dept}] :', best[1])
            return

        next_states = list(compute_available_next(current_state, system_states))

        for i, state in enumerate(progressbar(next_states, dept, tk)):
        # for i, state in enumerate(next_states):
            # print(f'{current_state} -> {state}')
            # Simulate the new state:
            new_cost = cost
            if current_state[0] != state[0]: # Load input dma
                new_cost += 1
            if current_state[1] != state[1]: # Load and store output dma
                new_cost += 2
            
            if new_cost > best[0]: # or new_cost > 10: # No the best
                return
            
            catched_index = state_eval_all[state[0]][state[1]]
            catched_value = tensor_o.ravel()[catched_index]
            is_not_match = np.all(catched_value == -1)
            
            if not is_not_match: # This state has interest (NOT TRUE, but speed up a lot)
                # Apply modifications
                next_tensor_o = np.copy(tensor_o) # We need to preserve the state
                history_stack[dept] = state # stack push
                next_tensor_o.ravel()[catched_index] = -1
                compute(state, system_states - {state}, next_tensor_o, new_cost, dept+1, history_stack, best, state_eval_all)

    tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
    tensor_o = toeplitz(tensor_i, Dky, Dkx)

    # Compute all combinations of @input, @output (we will use remove on it -> set)
    combination_dma_io = list(product(range(tensor_i.size - DMA + 1), range(tensor_o.size - DMA + 1)))
    combination_dma_io_valid = set()

    # Remove comb without interest
    for comb in combination_dma_io:
        tensor_i_dma = dma_load(tensor_i, comb[0])
        tensor_o_dma = dma_load(tensor_o, comb[1])
        diff = compare_dma(tensor_i_dma, tensor_o_dma)
        if diff.size:
            combination_dma_io_valid.add(comb)
    # print(list(combination_dma_io))
    
    def state_eval(comb):
        tensor_i_dma = dma_load(tensor_i, comb[0])
        tensor_o_dma = dma_load(tensor_o, comb[1])
        for idmai, vi in enumerate(tensor_i_dma):
            for idmao, vo in enumerate(tensor_o_dma):
                if vi == vo:
                    yield comb[1] + idmao

    with enlighten.Manager() as manager:
        history_stack = [None for _ in combination_dma_io_valid]
        state_eval_all = [[list(state_eval((i, j))) for j in range(tensor_o.size - DMA + 1)] for i in range(tensor_i.size - DMA + 1)]
        tk = progressbar_init(manager, ("loop0", 'loop1'))
        best = [999999, None]
        compute((-1, -1),                 # Do not force initial state
                combination_dma_io_valid, # All state te explore
                tensor_o,                 # Default result
                0, 0, history_stack, best, state_eval_all)
    return best

from copy import copy, deepcopy

def algo0FullExploreNumba(y, x, Dky, Dkx):
   
    @jit(nopython=True)
    def compute(current_state, system_states, system_states_iterator, system_states_iterator_row_col,
                tensor_o, cost, dept, history_stack, best_history_cost_dept, best_history_stack):

        if np.all(tensor_o == -1): # Done
            if cost < best_history_cost_dept[0]:
                best_history_cost_dept[0] = cost
                best_history_cost_dept[1] = dept
                best_history_stack[:dept] = history_stack[:dept]
                # print(f'HIT [{cost}/{dept}] :', best_history_stack[:dept])
            return

        # for i, state in enumerate(progressbar(next_states, dept, tk)):
        state_iterator = system_states_iterator if current_state[0] != -1 else system_states_iterator_row_col[current_state[0]][current_state[1]]

        
        for istate, state in enumerate(state_iterator):
            # if dept == 0:
            #     print(f"[{istate}/{len(state_iterator)}]")
    
            if system_states[state[0]][state[1]] == 0:
                continue

            # Evaluate cost:
            new_cost = cost
            if current_state[0] != state[0]: # Load input dma
                new_cost += 1
            if current_state[1] != state[1]: # Load and store output dma
                new_cost += 2
            
            if new_cost > best_history_cost_dept[0]: # or new_cost > 10: # No the best
                return          

            # Apply modifications
            system_states[state[0]][state[1]] = 0 # Visited
            next_tensor_o = np.copy(tensor_o) # We need to preserve the state
            history_stack[dept] = state # stack push
            for vi in range(state[0], state[0] + DMA):
                for itensoro in range(state[1], state[1] + DMA):
                    if vi == next_tensor_o.ravel()[itensoro]:
                        next_tensor_o.ravel()[itensoro] = -1

            compute(state, system_states, system_states_iterator, system_states_iterator_row_col,
                    next_tensor_o, new_cost, dept+1, history_stack, best_history_cost_dept, best_history_stack)
            
            # Restore modifications
            system_states[state[0]][state[1]] = 1 # Unvisited

    tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
    tensor_o = toeplitz(tensor_i, Dky, Dkx)

    def rowcol_iterator(i, j, tensor):
        for jj in range(tensor.shape[1]):
            if (j != jj):
                yield (i, jj)
        for ii in range(tensor.shape[0]):
            if (i != ii):
                yield (ii, j)
              
    nb_comb = (tensor_i.size - DMA + 1) * (tensor_o.size - DMA + 1)
    history_stack = np.zeros((nb_comb, 2), dtype=np.int32)
    best_history_stack = np.zeros((nb_comb, 2), dtype=np.int32) 
    best_history_cost_dept = np.array([999999, 0])

    system_states = np.ones((tensor_i.size - DMA + 1, tensor_o.size - DMA + 1), dtype=np.int32)
    print(system_states)
    system_states_iterator = np.array(list(product(range(system_states.shape[0]), range(system_states.shape[1]))), dtype=np.int32)
    print(f'{system_states_iterator.shape=} # {np.prod(system_states_iterator.shape)}')
    system_states_iterator_row_col = np.array([[list(rowcol_iterator(i, j, system_states)) for i in range(tensor_i.size - DMA + 1)] for j in range(tensor_o.size - DMA + 1)], dtype=np.int32)
    # (tensor_i.size - DMA + 1,
    #  tensor_o.size - DMA + 1, 
    #  tensor_i.size - DMA + 1 + tensor_o.size - DMA + 1 (-2),
    #  2)
    print(f'{system_states_iterator_row_col.shape=} # {np.prod(system_states_iterator_row_col.shape)}')
    
    compute(np.array([-1, -1]),                 # Do not force initial state
            system_states, system_states_iterator, system_states_iterator_row_col, # All state te explore
            tensor_o,                 # Default result
            0, 0, history_stack, best_history_cost_dept, best_history_stack)
    
    best = [best_history_cost_dept[0], best_history_stack[:best_history_cost_dept[1]]]

    return best

def algo1(y, x, Dky, Dkx):
    """ ALGO 1 """
    tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
    tensor_o = toeplitz(tensor_i, Dky, Dkx)

    
    # Compute all combinations of @input, @output (we will use remove on it -> set)
    combination_dma_io = list(product(range(tensor_i.size - DMA + 1), range(tensor_o.size - DMA + 1)))
    print(list(combination_dma_io))
    def state_eval(comb):
        tensor_i_dma = dma_load(tensor_i, comb[0])
        tensor_o_dma = dma_load(tensor_o, comb[1])
        for idmai, vi in enumerate(tensor_i_dma):
            for idmao, vo in enumerate(tensor_o_dma):
                if vi == vo:
                    yield comb[1] + idmao


    # for comb in combination_dma_io:
    #    print(f'{comb} -> {list(state_eval(comb))}')
    return 
    state_eval_all = [[set(state_eval((i, j))) for j in range(tensor_o.size - DMA + 1)] for i in range(tensor_i.size - DMA + 1)]

    
    index_to_states = [set() for _ in range(tensor_o.size)]
    for state in combination_dma_io:
        for v in state_eval(state):
            index_to_states[v].add(state)

    def explore(index_to_states, path, history, dept, tk):       

        def hist_is_prefix(path, history):
            for hist in history:
                is_prefix = True
                for h in history:
                    if not h in path:
                        is_prefix = False
                        break
                if is_prefix:
                    return True
                
            return False
                    
        # if hist_is_prefix(path, history):
        #     return
        
        valcount = np.array(list(map(len, index_to_states)))
        # print(valcount)
        if np.all(valcount == 0): # DONE
            print(f'HIT! : {dept} {path}')
            history.append(set(path))
            return

        # Find the essential states
        indx = np.where(valcount == 1)[0]
        if len(indx): # There is multiple values: only do the first
            indx = [indx[0]]
        else: # There is no values: fallback to counts != 0
            indx = np.where(valcount != 0)[0]
        
        def get_states(indx):
            for ind in indx:
                for sel_state in index_to_states[ind]:
                    yield sel_state
        
        for sel_state in progressbar(set(get_states(indx)), dept, tk):
                # print(ind, sel_state)
                index_to_states_new = deepcopy(index_to_states) # AYA 
                path.append(sel_state)
                # update state_result (remove catched values)
                for i in state_eval_all[sel_state[0]][sel_state[1]]:
                    index_to_states_new[i] = set()
                explore(index_to_states_new, path, history, dept+1, tk)
                path.pop()
    
    with enlighten.Manager() as manager:
        tk = progressbar_init(manager, ("loop0", 'loop1', 'loop3', 'loop4'))
        explore(index_to_states, [], [], 0, tk)


    # print(state_result)
    # valcount = np.zeros((tensor_o.size))
    # comefrom = [None for _ in range(tensor_o.size)]
    # for state in combination_dma_io:
    #     res = state_result[state[0]][state[1]]
    #     # print(f'{state} --> {res}')
    #     for v in res:
    #         comefrom[v] = state
    #     valcount[list(state_result[state[0]][state[1]])]+=1
    # print(valcount)
    # indx = np.where(valcount == 1)[0] # Find the essential states
    # print(indx)
    # sel_ind = indx[0]
    # print(f'{sel_ind=}')
    # sel_state = comefrom[sel_ind]
    # sel_res = copy(state_result[sel_state[0]][sel_state[1]])
    # # Find sel_ind in
    # # update state_result OUUTCH !!
    # for state in combination_dma_io:
    #     res = state_result[state[0]][state[1]]
    #     for v in sel_res:
    #         res -= {v}
    # print(state_result)


def algo2(y, x, Dky, Dkx):
    """ ALGO 2 """
    tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
    tensor_o = toeplitz(tensor_i, Dky, Dkx)

    
    # Compute all combinations of @input, @output (we will use remove on it -> set)
    combination_dma_io = list(product(range(tensor_i.size - DMA + 1), range(tensor_o.size - DMA + 1)))
    print(list(combination_dma_io))
    def state_eval(comb):
        tensor_i_dma = dma_load(tensor_i, comb[0])
        tensor_o_dma = dma_load(tensor_o, comb[1])
        for idmai, vi in enumerate(tensor_i_dma):
            for idmao, vo in enumerate(tensor_o_dma):
                if vi == vo:
                    yield comb[1] + idmao

    state_eval_all = [[set(state_eval((i, j))) for j in range(tensor_o.size - DMA + 1)] for i in range(tensor_i.size - DMA + 1)]
    for comb in combination_dma_io:
        print(f'{comb} -> {list(state_eval(comb))}')

    state_eval_i = [set() for _ in range(tensor_i.size - DMA + 1)]
    for i, s in enumerate(state_eval_i):
        for o in range(tensor_o.size - DMA + 1):
            for v in state_eval_all[i][o]:
                state_eval_i[i].add(v)
    
    for i, s in enumerate(state_eval_i):
        print(f'i:{i} -> {s}')

    state_eval_o = [set() for _ in range(tensor_o.size - DMA + 1)]
    for o, s in enumerate(state_eval_o):
        for i in range(tensor_i.size - DMA + 1):
            for v in state_eval_all[i][o]:
                state_eval_o[o].add(v)
    
    for o, s in enumerate(state_eval_o):
        print(f'o:{o} -> {s}')

    
def evaluate_prog(prog, input_size, output_size):
    mvcount = len(list(filter(lambda x:x[0] == 'mv', prog)))
    dmacount = len(prog) - mvcount
    instrcount = len(prog)
    
    count_ldi = len(list(filter(lambda x:x[0] == 'ldi', prog)))
    count_ldo = len(list(filter(lambda x:x[0] == 'ldo', prog)))
    count_sto = len(list(filter(lambda x:x[0] == 'sto', prog)))
    
    print(f'=========== EVALUATE PROG ===========')
    print(f'  total instructions : {instrcount}')
    print(f'           total ldi : {count_ldi}')
    print(f'           total ldo : {count_ldo}')
    print(f'           total sto : {count_sto}')
    print(f'        number of mv : {mvcount}')
    print(f' number of dma ld st : {dmacount}')
    print(f'   mean mv per ld/st : {mvcount/dmacount}')
    print(f'             quality : {(input_size+output_size)/(count_ldi+count_sto)/DMA}')
    print(f'         ldi_quality : {input_size/(count_ldi*DMA)}')
    print(f'         sto_quality : {output_size/(count_sto*DMA)}')
    

import time

best = [10, np.array([(4, 9), (4, 12), (4, 4), (0, 4), (0, 0)])]
tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
tensor_o = toeplitz(tensor_i, Dky, Dkx)

if 1:
    start_time = time.time()
    best = algo0(y, x, Dky, Dkx)
    print(best)
    prog = export_algo(best[1], tensor_o)
    evaluate_prog(prog, 9, 16)
    print("--- algo0 : %s seconds ---" % (time.time() - start_time))

if 1:
    start_time = time.time()
    best = algo0FullExploreNumba(y, x, Dky, Dkx)
    best[1] = list(map(tuple, best[1]))
    print(best)
    prog = export_algo(best[1], tensor_o)
    evaluate_prog(prog, 9, 16)
    print("--- algo0FullExploreNumba : %s seconds ---" % (time.time() - start_time))

# evaluate_prog(prog, 9, 16)

# algo1(y, x, Dky, Dkx)
# algo2(y, x, Dky, Dkx)
