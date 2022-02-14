"""
Analyse memory transaction during toeplitz transformation
"""

import numpy as np
import sys
import enlighten
from itertools import product
import time
from numba import jit, njit
from prog import Prog

def profile(func):
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()
    ret = func()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    return ret

# use 5 word DMA buffer
DMA = 16

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

def is_tensor_fully_catched(tensor):
    return np.all(tensor == -1)

# print(f'{dma_load(tensor_i, 0)=}')
# print(f'{dma_load(tensor_i, 4)=}')
# print(f'{dma_load(tensor_i, 15)=}')

# tensor_i_dma = dma_load(tensor_i, 0)
# tensor_o_dma = dma_load(tensor_o, 8)
# print(tensor_i_dma, tensor_o_dma)


# dma_match = compare_dma(tensor_i_dma, tensor_o_dma)
# print(f'{dma_match=}')

# print(tensor_o)
# print(tensor_i_dma, tensor_o_dma)

# is_tensor_fully_catched(tensor_o)
# print(tensor_o)


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


def algo0(y, x, Dky, Dkx):
    
    def compute(current_state, system_states, tensor_o, cost, dept, history_stack, best, state_eval_all):

        # Heuristique
        def compute_available_next(cur, combs):
            # return combs
            ret = None 
            if cur == (-1, -1):
                ret = combs
            else:
                ret = filter(lambda x: x[0] == cur[0] or x[1] == cur[1], combs)
            return ret


        def heuristique(states):
            states = list(states)
            max_add = lambda x: -np.sum(tensor_o.ravel()[state_eval_all[x[0]][x[1]]] != -1)
            states.sort(key = max_add)
            states = states[:max(3, len(states)//(dept+20))] # states[:max(3, len(states)//(dept+20))]
            return states

        if np.all(tensor_o == -1): # Done
            if cost < best[0]:
                best[0] = cost
                best[1] = history_stack[:dept]
                print(f'HIT [{cost}/{dept}] :', best[1])
            return

        next_states = heuristique(compute_available_next(current_state, system_states))

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
        tk = progressbar_init(manager, [f'l{k}' for k in range(10)])
        best = [999999, None]
        compute((-1, -1),                 # Do not force initial state
                combination_dma_io_valid, # All state te explore
                tensor_o,                 # Default result
                0, 0, history_stack, best, state_eval_all)
    return best

from copy import copy, deepcopy

def algo0FullExploreNumba(y, x, Dky, Dkx):
   
    @jit(nopython=True, parallel=True)
    def compute(current_state, system_states, system_states_iterator, system_states_iterator_row_col,
                tensor_o, cost, dept, history_stack, best_history_cost_dept, best_history_stack):

        if np.all(tensor_o == -1): # Done
            if cost < best_history_cost_dept[0]:
                best_history_cost_dept[0] = cost
                best_history_cost_dept[1] = dept
                best_history_stack[:dept] = history_stack[:dept]
                print(f'HIT [{cost}/{dept}] :', best_history_stack[:dept])
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

from numba_progress import ProgressBar

def algo1(y, x, Dky, Dkx, v_heuristic=False, 
                          v_fast_explore=False,
                          v_fast_explore_numba=False):
    """ ALGO 1 """

    if v_heuristic + v_fast_explore + v_fast_explore_numba != 1:
        raise Exception('Invalid arguments')

    tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
    tensor_o = toeplitz(tensor_i, Dky, Dkx)
    

    # print(list(combination_dma_io))
    def state_eval(comb):
        tensor_o_dma = dma_load(tensor_o, comb[1])
        for idmao, vo in enumerate(tensor_o_dma):
            if vo >= comb[0] and vo < comb[0] + DMA:
                yield comb[1] + idmao


    # for comb in combination_dma_io:
    #    print(f'{comb} -> {list(state_eval(comb))}')

   

    def explore(index_to_states, path, history, dept):       
        if dept >= history[0]:
            return

        valcount = np.array(list(map(len, index_to_states)))

        # print(valcount)
        if np.all(valcount == 0): # DONE
            if dept < history[0]:
                print(f'HIT! : {dept} {path}')
                history[0] = dept
                history[1] = np.copy(path)
            return

        # Find the essential states
        indx = np.where(valcount != 0)[0]

        def get_states(indx):
            global indd
            # Heuristic
            indx = list(indx)
            # indx.sort(key=lambda i: len(index_to_states[i]))
            indx = indx[:1] # Explone the most criticals index
            # indx = [min(indx, key=lambda i: len(index_to_states[i]))]
            # for ind in indx:
            for ind in indx: 
                indd = ind
                states = list(index_to_states[ind])
                # print(states)
                states.sort(key = lambda s: -np.sum([s in index_to_states[k] for k in range(tensor_o.size)]))
                states = states[:1] # Explone the most usefull states
                # states = [min(states, key=lambda s: -np.sum([s in index_to_states[k] for k in range(tensor_o.size)]))]
                for sel_state in states:
                    # print(f'{ind:5} {sel_state}')
                    yield sel_state

        for sel_state in progressbar(set(get_states(indx)), dept, tk):
            index_to_states_new = deepcopy(index_to_states) # AYA 
            path.append(sel_state)
            # update state_result (remove catched values)
            for i in state_eval_all[sel_state[0]][sel_state[1]]:
                index_to_states_new[i] = set()
            explore(index_to_states_new, path, history, dept+1)
            path.pop()
    

    def fast_explore():

        def get_states_at_index(ind_o):
            ind_i = tensor_o.ravel()[ind_o]
            polyhedron = product(range(max(0, ind_i - DMA + 1), min(ind_i + 1, tensor_i.size - DMA + 1)),
                                 range(max(0, ind_o - DMA + 1), min(ind_o + 1, tensor_o.size - DMA + 1)))
            return polyhedron

        def counte_states_as_index(ind_o):
            ind_i = tensor_o.ravel()[ind_o]
            return ((min(ind_i + 1, tensor_i.size - DMA + 1) -  max(0, ind_i - DMA + 1)) *
                    (min(ind_o + 1, tensor_o.size - DMA + 1) - max(0, ind_o - DMA + 1)))
 
        def test_if_index_match_state(ind_o, state):
            # EQ as state in get_states_at_index(ind_o)
            ind_i = tensor_o.ravel()[ind_o]
            return (state[0] >= max(0, ind_i - DMA + 1) and
                    state[0] < min(ind_i + 1, tensor_i.size - DMA + 1) and 
                    state[1] >= max(0, ind_o - DMA + 1) and
                    state[1] < min(ind_o + 1, tensor_o.size - DMA + 1))
    
        index_to_count_states = np.array([counte_states_as_index(ind_o) for ind_o in range(tensor_o.size)])
        # print(index_to_count_states)
        
        path = []
        dept = 0  
        
        pbar = enlighten.Counter(total=tensor_o.size, unit='ticks')
        while not np.all(index_to_count_states == 0):
            dept += 1
            # Explore the most critical index
            residual_indx = filter(lambda i: index_to_count_states[i] != 0, range(tensor_o.size))
            # ind = min(residual_indx, key=lambda i: index_to_count_states[i])
            ind = next(iter(residual_indx))
            # print("ind", ind)
            # Explone the most usefull state
            available_states = list(get_states_at_index(ind))
            # for s in available_states:
            #    print(s, -sum((index_to_count_states[k] != 0 and test_if_index_match_state(k, s) for k in range(tensor_o.size))))
            # print(available_states)
            sel_state = min(available_states, key = lambda s: -sum((index_to_count_states[k] != 0 and test_if_index_match_state(k, s) for k in range(tensor_o.size))))
            # print(f'{dept=} {ind=} {sel_state}')
            # print("sel_state", sel_state)
            path.append(sel_state)
            # update state_result (remove catched values)
            for i in state_eval(sel_state):
                pbar.update()
                index_to_count_states[i] = 0
            

        # print(f'HIT! : {dept} {path}')
        return [dept, path]

    @njit(nogil=True)
    def fast_explore_numba(tensor_i, tensor_o, history_stack, progress_proxy):
        
        def counte_states_as_index(tensor_i, tensor_o, ind_o):
            ind_i = tensor_o.ravel()[ind_o]
            return ((min(ind_i + 1, tensor_i.size - DMA + 1) -  max(0, ind_i - DMA + 1)) *
                    (min(ind_o + 1, tensor_o.size - DMA + 1) - max(0, ind_o - DMA + 1)))
    
        index_to_count_states = np.array([counte_states_as_index(tensor_i, tensor_o, ind_o) for ind_o in range(tensor_o.size)])
        # print(index_to_count_states)
        
        dept = 0  
        while not np.all(index_to_count_states == 0):
            # Explore the most critical index
            vmin = np.inf
            ind = 0
            # No need to filter !
            # for i in range(tensor_o.size):
            #     if index_to_count_states[i] != 0:
            #         if index_to_count_states[i] < vmin:
            #             ind = i
            #             vmin = index_to_count_states[i]
            for i in range(tensor_o.size):
                if index_to_count_states[i] != 0:
                    ind = i
                    break
            # print(ind)
            # Explore the most usefull state
            sel_state = (0, 0)
            vmin = np.inf
            ind_o = ind
            ind_i = tensor_o.ravel()[ind_o]
            for i in range(max(0, ind_i - DMA + 1), min(ind_i + 1, tensor_i.size - DMA + 1)):
                for o in range(max(0, ind_o - DMA + 1), min(ind_o + 1, tensor_o.size - DMA + 1)):
                    # available_states
                    test = 0
                    for k in range(o, o + DMA - 1 + 1): # We can only hit here 
                        # +1 for range; -1 for loop o >= max(0, k - DMA + 1)
                        loc_ind_i = tensor_o.ravel()[k]
                        test -= index_to_count_states[k] != 0 and (i >= loc_ind_i - DMA + 1 and i < loc_ind_i + 1)
                        # 
                        # i >= max(0, loc_ind_i - DMA + 1) and
                        # i < min(loc_ind_i + 1, tensor_i.size - DMA + 1) and 
                        # o >= max(0, k - DMA + 1) and
                        # o < min(k + 1, tensor_o.size - DMA + 1))

                    # print((i, o), test)
                    if test < vmin:
                        vmin = test
                        sel_state = (i, o)

            # print('sel_state', sel_state)
            history_stack[dept] = sel_state
            
            # update state_result (remove catched values)
            for io in range(sel_state[1], sel_state[1] + DMA):
                vo = tensor_o.ravel()[io]
                if vo >= sel_state[0] and vo < sel_state[0] + DMA:
                    i = io
                    progress_proxy.update(1)
                    # # pbar.update()
                    index_to_count_states[i] = 0
        
            dept += 1

        # print(f'HIT! : {dept} {path}')
        return dept

    def tsp_solve(states):
        """
        Find the best sheduling of states
        We solve TSP with Christofides algorithm
        """
        def distance(state0, state1):
            dist = 0
            if state0[0] != state1[0]:
                dist += 1
            if state0[1] != state1[1]:
                dist += 2
            return dist

        from python_tsp.exact import solve_tsp_dynamic_programming
        from python_tsp.heuristics import solve_tsp_local_search

        distances = np.array([[distance(s0, s1) for s1 in states] for s0 in states], dtype=np.float16)
        # print(distances)
        permutation, cost = solve_tsp_local_search(distances)
        return [cost, np.array(states)[permutation]]

    if v_heuristic: # Generic version
        print('================== generic version ======================')
        start_time = time.time()
        with enlighten.Manager() as manager:
             # Compute all combinations of @input, @output (we will use remove on it -> set)
            print("Compute combination_dma_io ...", end='', flush=True)
            combination_dma_io = list(product(range(tensor_i.size - DMA + 1), range(tensor_o.size - DMA + 1)))
            print(f'DONE #comb = {len(combination_dma_io)}')   
            print("Compute index_to_states ...", end='', flush=True)
            index_to_states = [[] for _ in range(tensor_o.size)]
            pbar = enlighten.Counter(total=len(combination_dma_io), unit='ticks')
            for state in combination_dma_io:
                pbar.update()
                for v in state_eval(state):
                    index_to_states[v].append(state)
            print('DONE')
            print("Compute state_eval_all ...", end='', flush=True)
            state_eval_all = [[set(state_eval((i, j))) for j in range(tensor_o.size - DMA + 1)] for i in range(tensor_i.size - DMA + 1)]
            print('DONE')
            best = [99999, None]
            tk = progressbar_init(manager, [f"l{k}" for k in range(10)])
            explore(index_to_states, [], best, 0)
        print(f"Got states: {list(map(tuple, best[1]))}")
        print(f"Unoptimized cost: {3*best[0]}")
        print("--- basis explore : %s seconds ---" % (time.time() - start_time))
        print('==========================================================')
    if v_fast_explore: # fast_explore
        print('================== fast_explore ==========================')
        start_time = time.time()
        best = fast_explore()
        print(f"Got states: {list(map(tuple, best[1]))}")
        print(f"Unoptimized cost: {3*best[0]}")
        print("--- fast_explore : %s seconds ---" % (time.time() - start_time))
        print('==========================================================')
    if v_fast_explore_numba: # fast_explore_numba
        print('================== fast_explore_numba ====================')
        start_time = time.time()
        history_stack = np.zeros((tensor_o.size, 2), dtype=np.int32)
        with ProgressBar(total=tensor_o.size) as progress:
            dept = fast_explore_numba(tensor_i, tensor_o, history_stack, progress)
        best = [dept, history_stack[:dept]]
        print(f"Got states: {list(map(tuple, best[1]))}")
        print(f"Unoptimized cost: {3*best[0]}")
        print("--- fast_explore_numba : %s seconds ---" % (time.time() - start_time))
        print('==========================================================')
    
    best = tsp_solve(best[1])
    # print(f"Got states: {list(map(tuple, best[1]))}")
    # print(f"cost: {best[0]}")
    return best

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


def export(states, tensor_o):
    state = [-1, -1]
    prog = Prog()

    def transactions(tensor_o):
        dmai = dma_load(tensor_i, state[0])
        dmao = dma_load(tensor_o, state[1])
        for idmai, vi in enumerate(dmai):
            for idmao, vo in enumerate(dmao):
                if vi == vo:
                    dmao[idmao] = -1 # In reality, copy value:
                    prog.append_mv(idmao, idmai)
        return prog

    for i, o in states:
        if state[0] != i:
            prog.append_ldi(i)
            # prog += f'DMA_LD(dma_i, {i})\n'
            state[0] = i
            transactions(tensor_o)

        if state[1] != o:
            if state[1] != -1: # Small optim
                prog.append_sto(state[1])
                # prog += f'DMA_ST(dma_o, {state[1]})\n'
            prog.append_ldo(o)
            # prog += f'DMA_LD(dma_o, {o})\n'
            state[1] = o
            transactions(tensor_o)
    
    prog.append_sto(o)
    assert np.all(tensor_o == -1)
    return prog



# Input
x = 8
y = 8

# Filter shape
Dkx = 2
Dky = 2

best = [10, np.array([(4, 9), (4, 12), (4, 4), (0, 4), (0, 0)])]
tensor_i = np.arange(x*y, dtype=np.int32).reshape(y, x) # tab[index] = index !!
tensor_o = toeplitz(tensor_i, Dky, Dkx)

print(tensor_i)
print(tensor_o)

if 0:
    start_time = time.time()
    best = algo0(y, x, Dky, Dkx)
    print(best)
    prog = export(best[1], tensor_o)
    prog.evaluate(DMA)
    print("--- algo0 : %s seconds ---" % (time.time() - start_time))

if 0:
    start_time = time.time()
    best = algo0FullExploreNumba(y, x, Dky, Dkx)
    best[1] = list(map(tuple, best[1]))
    print(best)
    prog = export(best[1], tensor_o)
    prog.evaluate(DMA)
    print("--- algo0FullExploreNumba : %s seconds ---" % (time.time() - start_time))


if 1:
    start_time = time.time()
    best = algo1(y, x, Dky, Dkx, v_fast_explore_numba=True)
    prog = export(best[1], tensor_o)
    prog.evaluate(DMA)
    print("--- algo1 : %s seconds ---" % (time.time() - start_time))




# algo2(y, x, Dky, Dkx)
