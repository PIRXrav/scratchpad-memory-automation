import numpy as np
import sys
import enlighten
from itertools import product
import time
from numba import jit, njit
from prog import Prog


@njit(nogil=True)
def fast_explore_numba(tensor_i, tensor_o, dma, block_size, history_stack):

    def counte_states_as_index(tensor_i, tensor_o, ind_o):
        ind_i = tensor_o.ravel()[ind_o]
        return ((min(ind_i + 1, tensor_i.size - dma + 1) - max(0, ind_i - dma + 1)) *
                (min(ind_o + 1, tensor_o.size - dma + 1) - max(0, ind_o - dma + 1)))

    index_to_count_states = np.array([counte_states_as_index(tensor_i, tensor_o, i) for i in range(tensor_o.size)])
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
        # print('ind=', ind)
        # Explore the most usefull state
        sel_state = (0, 0)
        vmin = np.inf
        ind_o = ind
        ind_i = tensor_o.ravel()[ind_o]
        for i in range(max(0, ind_i - dma + 1), min(ind_i + 1, tensor_i.size)):  # tensor_i.size - dma + 1
            for o in range(max(0, ind_o - dma + 1), min(ind_o + 1, tensor_o.size)):  # tensor_o.size - dma + 1
                if i % block_size != 0:
                    continue
                if o % block_size != 0:
                    continue
                # available_states
                test = 0
                for k in range(o, min(o + dma - 1 + 1, tensor_o.size)):  # We can only hit here
                    # +1 for range; -1 for loop o >= max(0, k - DMA + 1)
                    loc_ind_i = tensor_o.ravel()[k]
                    test -= index_to_count_states[k] != 0 and (i >= loc_ind_i - dma + 1 and i < loc_ind_i + 1)
                    # i >= max(0, loc_ind_i - DMA + 1) and
                    # i < min(loc_ind_i + 1, tensor_i.size - DMA + 1) and
                    # o >= max(0, k - DMA + 1) and
                    # o < min(k + 1, tensor_o.size - DMA + 1))

                # print((i, o), test)
                if test < vmin:
                    vmin = test
                    sel_state = (i, o)

        if vmin == np.inf:
            raise Exception("No states !")

        history_stack[dept] = sel_state

        # update state_result (remove catched values)
        for io in range(sel_state[1], min(sel_state[1] + dma, tensor_o.size)):
            vo = tensor_o.ravel()[io]
            if vo >= sel_state[0] and vo < sel_state[0] + dma:
                index_to_count_states[io] = 0
        # print('sel_state', sel_state)
        # print('index_to_count_states', index_to_count_states)
        # print('history_stack', history_stack)
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

def run(tensor_i, tensor_o, dma, block_size, do_tsp=False):
    print('================== fast_explore_numba ====================(v2)')
    print(f'{tensor_i.size=} {tensor_o.size=} {dma=} {block_size=}')
    start_time = time.time()
    history_stack = np.zeros((tensor_o.size, 2), dtype=np.int32)
    dept = fast_explore_numba(tensor_i, tensor_o, dma, block_size, history_stack)
    best = [dept, history_stack[:dept]]
    print(f"Got states: {list(map(tuple, best[1]))}")
    print(f"Unoptimized cost: {3*best[0]}")
    print("--- fast_explore_numba : %s seconds ---" % (time.time() - start_time))
    print('==========================================================')
    print(best)
    if(do_tsp):
        best = tsp_solve(best[1])
    print(f"Got states: {list(map(tuple, best[1]))}")
    print(f"cost: {best[0]}")
    return best
