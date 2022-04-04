"""
SCRATCHPAD MEMORY AUTOMATION


C src
[pycparser] ---> pycparse AST
[AstToolkit] ---> Simplified AST (ast with only for/refs)
[__SMA__] ----> pycparse AST with memory transaction
[EXPORT] ----> C src

"""

from operator import index
import sys

from pycparser import parse_file, c_generator, CParser
from pycparser import c_ast

from collections import defaultdict
from itertools import chain


from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import TerminalFormatter

import numpy as np

import asttools as at
from asttools import stmt_c_to_ast, expr_c_to_ast

from math import ceil, floor

import logging

log = logging.getLogger(__name__)


DMA_SIZE = 129

class Gencode:
    @classmethod
    def cgen_dma_ld(self, adr, buff, size):
        return f"DMA_LD({adr}, {buff}, {size})"

    @classmethod
    def cgen_dma_st(self, adr, buff, size):
        return f"DMA_ST({adr}, {buff}, {size})"
    
    @classmethod
    def cgen_static_mac(self, A, B):
        return '+'.join(f'(({a}) * ({b}))' for a, b in zip(A, B))


import sys


def do_memory_mapping(ast, poly_decl_namespace):
    log.debug(f"{poly_decl_namespace=}")
    for topfor in at.c_ast_get_all_topfor(ast):
        do_memory_mapping_on_topfor(ast, topfor, poly_decl_namespace)


def do_memory_mapping_on_topfor(ast, topfor, poly_decl_namespace):
    log.debug("TOP FORS:")
    # print(at.ast_to_c_highlight(topfor))
    refs = at.c_ast_get_all_top_ref(topfor)
    nb_refs = len(refs)
    log.debug(f"TOP REFS ({nb_refs}):")
    for i, ref in enumerate((refs)):
        log.debug(f"{at.ast_to_c(ref):20} RW={at.c_ast_ref_is_rw(ast, ref)}")
        dma_mapping_algo3(ast, ref, i, poly_decl_namespace)

from copy import copy
import polyhedron as poly

def c_ast_to_interval(ref, namespace, intervalbegin0=False):
    """
    Analyse the reference
    """
    class RefVisitor(c_ast.NodeVisitor):
        """
        ArrayRef(name=ArrayRef(name=ID(name='tab0'
                                    ),
                            subscript=ID(name='j'
                                            )
                            ),
                subscript=ID(name='i'
                            )
        )
        return ['tab0', 'j', 'i']
        """

        def __init__(self):
            self.stack = []
            self.depsid = set()

        def generic_visit(self, node):
            raise Exception("Unsupported OP: ", node)

        def visit_ID(self, node):
            if node.name in namespace:
                interval = copy(namespace[node.name])
                self.depsid.add(node.name)
            else:
                raise Exception(f"Unknown ID: {node}")
            # print(f'Visit ID {node.name}: push({interval})')
            self.stack.append(interval)

        def visit_BinaryOp(self, node):
            for n in node:
                self.visit(n)
            b = self.stack.pop()
            a = self.stack.pop()
            if node.op == '+':
                interval = a + b
            elif node.op == '-':
                interval = a - b
            elif node.op == '/':
                interval = a / b
            else:
                raise Exception(f"Invalid OP {node.op} in {node}")
            # print(f'visit_BinaryOp : push({interval})')
            self.stack.append(interval)

            
        def visit_Constant(self, node):
            if node.type == 'int':
                v = int(node.value)
                interval = poly.Interval(0 if intervalbegin0 else v, v)
            else:
                raise Exception(f"Invalid type {node.type} in {node}")
            # print(f'visit_Constant : push({interval})')
            self.stack.append(interval)


    rv = RefVisitor()
    rv.visit(ref)
    interval = rv.stack.pop()
    if rv.stack != []:
        raise Exception(f"Error during visit: {stack=}")
    return interval, rv.depsid


def c_ast_ref_to_interval(ref, namespace):
    name, asts = at.c_ast_ref_get_l(ref)
    poly_ref = []
    for ast in asts:
        interval, deps = c_ast_to_interval(ast, namespace)
        # print(f"!!! INT({at.ast_to_c(ast)}) -> {interval} @ {namespace}")
        poly_ref.append((interval, deps))

    # print("!!! ", name, poly_ref)
    return name, asts, poly_ref


def c_ast_arraydecl_to_intervals(decl):
    name, type, asts = at.c_ast_arraydecl_get_l(decl)
    poly_decl = []
    for ast in asts:
        # Evaluate decl access without namespace
        interval, _ = c_ast_to_interval(ast, {}, intervalbegin0=True)
        poly_decl.append(interval)

    return name, asts, poly_decl 

def c_ast_loop_to_interval_name(for_nodes):
    # Comute L ast for all for nodes
    poly_loop = []
    for name, ast_a, ast_b in map(at.c_ast_for_get_l, for_nodes):
        a = c_ast_to_interval(ast_a, {})[0].b
        b = c_ast_to_interval(ast_b, {})[0].b
        poly_loop.append((poly.Interval(a, b), name))

    return poly_loop

def contain_ordered(listin, data):
    if listin == []:
        return True
    if data == []:
        return False
    if listin[0] == data[0]:
        return contain_ordered(listin[1:], data[1:])
    else:
        return contain_ordered(listin, data[1:])

 # Fake it with poly
def set_to_1(s): 
    """Temporary function
    """
    v = list(s)
    res = v.pop()
    if v != []:
        raise Exception("Invalid set:", s)
    return res

import sympy

def dma_mapping_algo3(ast, ref, iref, poly_decl_namespace):
    """ """
    log.debug(f'========== DMA MAPPING {at.ast_to_c(ref)}')

    # Analyse Loops
    for_nodes = at.c_ast_get_for_fathers(ast, ref)
    poly_loop = c_ast_loop_to_interval_name(for_nodes)
    poly_loop_namespace = {name: interval for interval, name in poly_loop}

    # Analyse reference
    ref_is_read, ref_is_write = at.c_ast_ref_is_rw(ast, ref)
    ref_name, ref_l_ast, poly_ref = c_ast_ref_to_interval(ref, namespace=poly_loop_namespace)
    poly_ref_deps = [names for _, names in poly_ref]
    poly_ref_all_deps = set(chain(*poly_ref_deps))
    ref_access_names = [at.ast_to_c(ast) for ast in ref_l_ast]

    # Analyse reference declaration
    poly_decl = poly_decl_namespace[ref_name]
    poly_decl = list(reversed(poly_decl))
    ref_decl_l = [interval.area() for interval in poly_decl]
    ref_decl_l_cum = list(np.cumprod(ref_decl_l))
    
    # Gencode vars
    loops_access_l = [interval.area() for interval, _ in poly_loop]
    loops_access_names = [name for _, name in poly_loop]

    # for (int n = 0; n < 1000; n++){ # Cost 
    #     for (int i = 0; i < 100; i++){ # Cost 110
    #         for(int j = 0; j < 10; j++{ # Cost 10
    #             tab[n + i + i + 20] 
    #         }
    #     }
    # }
    # Loop   : i     -> out_name, Interval
    # ref    : i     -> equation, in_names
    # decl   : i     -> Interval

    # Loop = <i->name, IPoly>
    # Ref  = <names->IPoly>
    
    # Interval = [a, b]
    # NPoly   : i    -> name, Interval
    # IPoly   : i    -> Interval
    # PrePoly : i    -> in_names, equation
    # PrePoly.evaluate(namespace=NPoly) -> iPoly
    # 
    # IPoly.divise_by(DMA) -> #sub_poly, IR, ...
    # 

    # if 'input' in ref_name:
    #    return
    # if 'weights' in ref_name:
    #     print("-- no weights")
    #     return
    # if 'out' in ref_name:
    #     print("-- no out")
    #     return

    # Compute poly ref area
    from copy import deepcopy
    poly_loop_namespace_sparse = deepcopy(poly_loop_namespace)
    loops_ref_access_l_cum = []
    for name in ("", * reversed(loops_access_names[1:])):
        # Compute poly with cuts
        poly_loop_namespace_sparse[name] = poly.Interval(0, 0)
        _, _, pr = c_ast_ref_to_interval(ref, namespace=poly_loop_namespace_sparse)
        # Compute poly area
        area = 1
        for ir, (interval, deps) in enumerate(reversed(pr)):
            v = interval.area()
            if v:
                area *= v
                ir = len(ref_decl_l_cum) - ir -1
                if ir != 0:
                    area *= ref_decl_l_cum[ir-1]
                break

        # Append area
        loops_ref_access_l_cum.insert(0, area) 
    
    loops_ref_access_l_ref_expr_dma = [[sympy.parsing.sympy_parser.parse_expr(s, evaluate=False) for s in (ref_access_names)]]
    for name in reversed(loops_access_names):
        # TODO use sympy
        cur_exprs = deepcopy(loops_ref_access_l_ref_expr_dma[-1])
        for i, _ in enumerate(cur_exprs):
            cur_exprs[i] = cur_exprs[i].subs(name, 0)
        loops_ref_access_l_ref_expr_dma.append(cur_exprs)
    loops_ref_access_l_ref_expr_dma = list(reversed(loops_ref_access_l_ref_expr_dma))

    loops_ref_access_l_ref_expr_ram = [[sympy.parsing.sympy_parser.parse_expr(s, evaluate=False) for s in (ref_access_names)]]
    for name in loops_access_names:
        cur_exprs = deepcopy(loops_ref_access_l_ref_expr_ram[-1])
        for i, _ in enumerate(cur_exprs):
            cur_exprs[i] = cur_exprs[i].subs(name, 0)
        loops_ref_access_l_ref_expr_ram.append(cur_exprs)
   
    # loops_ref_access_l_ref_expr.pop(0) # Remove sentinel

    
    log.debug(f"{ref_name=}")
    log.debug(f"{ref_is_read=}")
    log.debug(f"{ref_is_write=}")
    log.debug(f"{poly_decl=}")
    log.debug(f"  |->{ref_decl_l=}")
    log.debug(f"  '->{ref_decl_l_cum=}")
    log.debug(f"{poly_loop=}")
    log.debug(f"  |->{loops_access_names=}")
    log.debug(f"  |->{loops_access_l=}")
    log.debug(f"  `->{poly_loop_namespace=}")
    log.debug(f"{poly_ref=}")
    log.debug(f"  |->{ref_access_names=}")
    log.debug(f"  |->{poly_ref_deps=}")
    log.debug(f"  |->{poly_ref_all_deps=}")
    log.debug(f"LOOP X REF")
    log.debug(f"  `->{loops_ref_access_l_cum=}")
    log.debug(f"  `->{loops_ref_access_l_ref_expr_dma=}")
    log.debug(f"  `->{loops_ref_access_l_ref_expr_ram=}")


    # Find where insert DMA LD/ST
    IL = 0
    if DMA_SIZE >= loops_ref_access_l_cum[-1]:  # We only need to do 1 transfert
        IL = -1
    else:
        while DMA_SIZE > loops_ref_access_l_cum[IL]:
            IL += 1


    # Find how to insert DMA LD/ST
    IR = 0
    if DMA_SIZE >= ref_decl_l_cum[-1]:  # We only need to do 1 transfert
        IR = -1
    else:
        while DMA_SIZE > ref_decl_l_cum[IR]:
            IR += 1

    log.debug(f"{IL=}")
    log.debug(f"{IR=}")

    log.debug(f"  `->{loops_ref_access_l_cum[IL-1]=}")
    log.debug(f"  `->{ref_access_names[0:IR]=}")

    # intervals_with_one_cut : {[a1, b1] , ..., [an, bn]} -> 
    # names_loop_intervals = CST
    # f IN intervals_with_one_cut(names_loop_intervals)
    # names_intervals = f(names_loop_intervals)
    # indx_interval = EVALUATE(ref_acces, names_intervals)
    # sizes     = CST
    # cost = ARRAY_C_USAGE(indx_interval, sizes)
    # cost < DMA
    # ==> Maximise cost !

    # ===> (NO)


    # poly_max = poly_loop # No cut
    # poly_min = poly.Polyhedron(1, 1)

    # INFINITY = DMA_SIZE + 1
    # def loss_function(poly):
    #     """The cost should be as close to DMA_SIZE as possible, but never higher
    #     """
    #     memory_cost_bytes = mem_cost(poly)
    #     if memory_cost_bytes > DMA_SIZE:
    #         return INFINITY
    #     else:
    #         return DMA_SIZE - memory_cost_bytes
    


    # while poly_max != poly_min:

    #     poly_mean = poly.Polyhedron.mean(poly_max, poly_min)

    #     loss_max  = loss_function(poly_max)
    #     loss_mean = loss_function(poly_mean)
    #     loss_min  = loss_function(poly_min)
        
    #     if loss_mean == INFINITY:
    #         poly_max = poly_mean
    #     else:
    #         poly_min = poly_mean



    # Remove __SMA__
    # for i, name in reversed(list(enumerate(loops_access_names))):
    #     if "__SMA__" in name:    
    #         loops_access_names.pop(i)
    #         loops_access_l.pop(i)
    




    # TODO: Lecture ne correspondant pas aux index direct exemple tab[i+1] (wip)
    # TODO: Partitionnement mémoire. Exemple Toeplitz matrix.
    # https://www.rle.mit.edu/eems/wp-content/uploads/2019/06/Tutorial-on-DNN-04-Kernel-Computations.pdf
    # Slide 25


    if IR == -1: # Array < DMA
        assert IR == IL
    elif IR == 0: # Divise
        pass
    else: # Repeat
        pass 

    current_name  = for_nodes[IL].init.decls[0].name

    buffer_name = f"__SMA__dma{iref}"
    adr_name = f"__SMA__{current_name}_adr{iref}"
    size_name = f"__SMA__{current_name}_size{iref}"
    iter_name = f"__SMA__{current_name}_i{iref}"
    cgen_dma_args = (adr_name, buffer_name, size_name)


    if IL == -1:
        topcomp = at.c_ast_get_upper_node(ast, for_nodes[-1])
    else:
        topcomp = for_nodes[IL].stmt

    if not topcomp:
        raise Exception(f"No {topcomp=} @ {IL=}")
    
    if IR == -1: # Array < DMA
        log.debug('--- WRAP MODE')
        # Compute memory mapping
        dma_transfer_size = loops_ref_access_l_cum[-1]
        dma_transfer_size_eff = dma_transfer_size
        tab_rw = ref_name
        log.debug(f"substitute {(tab_rw)} # mapped @ {buffer_name}")
        # Insert transactions
        topcomp.block_items.insert(0, stmt_c_to_ast(f"int {size_name} = {dma_transfer_size};"))
        topcomp.block_items.insert(1, stmt_c_to_ast(f'void * {adr_name} = {tab_rw};'))
        if ref_is_read:  # insert LD
            topcomp.block_items.insert(2, expr_c_to_ast(Gencode.cgen_dma_ld(*cgen_dma_args)))
        if ref_is_write:  # Insert ST
            topcomp.block_items.append(expr_c_to_ast(Gencode.cgen_dma_st(*cgen_dma_args)))
        # Update ref
        buff_adr = Gencode.cgen_static_mac(ref_access_names, [1, *ref_decl_l_cum[0:-1]])
        ast_buff = expr_c_to_ast(f"{buffer_name}[{buff_adr}]")
        at.c_ast_ref_update(ref, ast_buff.name, ast_buff.subscript)
        
    else:
        if IR == 0: # Divise
            nb_repeat = ref_decl_l[0] / DMA_SIZE
            nb_repeat_int = ceil(nb_repeat)
            dma_transfer_size = DMA_SIZE
            nb_repeat_residual = ref_decl_l[0] % DMA_SIZE
            dma_transfer_size_eff = DMA_SIZE
        else: # Repeat
            log.debug(f"we have to repeat {loops_access_l[IL]} time the loop '{loops_access_names[IL]}' of size {loops_ref_access_l_cum[IL - 1]}")
            # Find the best division
            # TODO: Algo: binary search for perfs

            def namespace_reduction(poly_loop_namespace, names, div_name, div_blocks):
                # Compute namespace
                poly_loop_namespace_partionned = deepcopy(poly_loop_namespace)
                for name in reversed(names[1:]):
                    if name == div_name:
                        poly_loop_namespace_partionned[name] = poly.Interval(0, div_blocks-1)
                        break
                    else:
                        poly_loop_namespace_partionned[name] = poly.Interval(0, 0)
                return poly_loop_namespace_partionned
    
            def compute_area(ref, poly_loop_namespace_partionned):
                # Compute memory footprint for this namespace
                _, _, pr = c_ast_ref_to_interval(ref, namespace=poly_loop_namespace_partionned)
                # Compute poly area
                area = 1
                for ir, (interval, deps) in enumerate(reversed(pr)):
                    v = interval.area()
                    if v:
                        area *= v
                        ir = len(ref_decl_l_cum) - ir -1
                        if ir != 0:
                            area *= ref_decl_l_cum[ir-1]
                        break
                return area
                

            for nb_block in reversed(range(1, loops_access_l[IL]+1+1)):
                poly_loop_namespace_partionned = namespace_reduction(
                    poly_loop_namespace,
                    loops_access_names, 
                    loops_access_names[IL], 
                    nb_block)
                area = compute_area(ref, poly_loop_namespace_partionned)
                # log.debug(f"{nb_block=} -> {area=}")
                # Is area valid AND is multiplicity ok ? and nb_repeat_residual == 0
                if area <= DMA_SIZE:
                    break
            
            log.debug(f"BEST: {nb_block=} => {area=}")

            # Number of repeat for this configuration
            nb_repeat = nb_block
            nb_repeat_int = nb_block
            nb_repeat_residual = loops_access_l[IL] % nb_block
            dma_transfer_size_eff = nb_repeat_int * loops_ref_access_l_cum[IL - 1]
            dma_transfer_size = area

            nb_repeat_block = floor(loops_access_l[IL] / nb_block)

            poly_loop_namespace_residual = namespace_reduction(
                poly_loop_namespace,
                loops_access_names, 
                loops_access_names[IL], 
                nb_repeat_residual)
            dma_transfer_size_residual = compute_area(ref, poly_loop_namespace_residual)

            log.debug(f" ===> DMA OPS: {nb_repeat_int}->{dma_transfer_size} & {nb_repeat_residual}->{dma_transfer_size_residual}")
            log.debug(f" ===> DMA OPS: {nb_repeat_block} x {dma_transfer_size} + {1} x {dma_transfer_size_residual}")
            
        log.debug('--- ' + ('DIVISE' if IR == 0 else 'REPEAT') + ' MODE')
        log.debug(f"{nb_repeat=}, {nb_repeat_int=}, {nb_repeat_residual=}")

        # Find the for @ IL
        
        ast_sub_for = c_ast.Compound(topcomp.block_items)
        # print(at.ast_to_c_highlight(ast_sub_for))

        # print(ast_to_c_highlight(ast_sub_for))

        # replace tab <-> BUFF
        # ref_access_names = loops_ref_access_l_ref_expr[IL - 1] # Get valid names
        # TODO outdated (/!\ smaller cube)
        if IR == 0: # Divise
            buff_adr = iter_name
        else: # Repeat
            buff_adr = Gencode.cgen_static_mac(loops_ref_access_l_ref_expr_dma[IL], [1, *ref_decl_l_cum[0:IR]]) + f'+ {iter_name}*{ref_decl_l_cum[IR-1]}'
                                               

        ast_buff = expr_c_to_ast(f"{buffer_name}[{buff_adr}]")
        at.c_ast_ref_update(ref, ast_buff.name, ast_buff.subscript)

        rn = loops_ref_access_l_ref_expr_ram[IL]
        tab_rw = ref_name + "".join(reversed(list((f"[{i}]" for i in rn))))
        log.debug(f"substitute {(tab_rw)} # mapped @ {buffer_name}s")
        
        body_repeat = DMA_SIZE if IR == 0 else nb_repeat_int

            
        stmts = []
        stmts.append(stmt_c_to_ast(f"static int {iter_name};"))
        stmts.append(stmt_c_to_ast(f"static int {size_name};"))
        stmts.append(stmt_c_to_ast(f'static void * {adr_name};'))

        
        if IR == 0: # Divise
            if nb_repeat_residual:
                size = f"MIN({dma_transfer_size}, ({loops_access_l[IL]}-{loops_access_names[IL]}))"
            else:
                size = str(DMA_SIZE)
        else: # Repeat
            # TODO: prevent memory overflow: DONE

            size = f"{loops_access_names[IL]} != {nb_repeat_block} * {nb_block} ? {dma_transfer_size} : {dma_transfer_size_residual}"
            # size = f"MIN({dma_transfer_size}, ({loops_access_l[IL]}-{loops_access_names[IL]})*{loops_ref_access_l_cum[IL-1]})"


        stmts.append(stmt_c_to_ast(f"if ({current_name} % {body_repeat} == 0) {{{iter_name} = 0; {size_name} = {size}; {adr_name} = {'&' + tab_rw};}}"))
        if ref_is_read:
            stmts.append(stmt_c_to_ast(f"if ({current_name} % {body_repeat} == 0) {{{Gencode.cgen_dma_ld(*cgen_dma_args)};}}"))
        
        for stmt in ast_sub_for.block_items:
            stmts.append(stmt)

        stmts.append(stmt_c_to_ast(f'{iter_name}++;'))

        if ref_is_write:
            stmts.append(stmt_c_to_ast(f"if ({current_name} % {body_repeat} == {body_repeat}-1 || {current_name} == {loops_access_l[IL]}-1) {{{Gencode.cgen_dma_st(*cgen_dma_args)};}}"))


        # print(ast_to_c_highlight(ast_intermediate))
        topcomp.block_items = stmts
    
    dma_efficiency = dma_transfer_size / DMA_SIZE
    log.debug(f"             {DMA_SIZE=}")
    log.debug(f"    {dma_transfer_size=}")
    log.debug(f"{dma_transfer_size_eff=}")
    efficiency = dma_transfer_size_eff/dma_transfer_size
    log.debug(f" ------>      DMA USAGE = {dma_transfer_size/DMA_SIZE*100}%")
    log.debug(f" ------> DMA EFFICIENCY = {efficiency*100}%")
    EFF_THRESHOLD = 0.1
    if (efficiency < EFF_THRESHOLD): # less than 1 ochet used for 10 loaded from memory
        log.warning(f"!!! {efficiency=} < {EFF_THRESHOLD} !!! You must do coalescing")
    log.debug(at.ast_to_c_highlight(ast))
    return dma_efficiency


# On veux 3 indicateurs:
#    - CPU   : l'algo ajoute peu de cycles cpu.
#    - DEBIT : les transferts mémoires sont de la taille du DMA au mieux.
#              On veux amortir le coup fixe
#    - PROG  : l'algo ne doit pas trop augmenter la taille du programme (bin)
#
# 3 optim possible:
# 1) Insertion entre les for.
#    - +CPU+PROG / -DEBIT (Sauf si on a de la chance)
#    - On minimise la taille du code
#    - Bien si taille proche de DMAMAX
# 2) Insertion dans le for avec un if.
#    - +DEBIT+PROG / -CPU
#    - On maximise le débit RAM
#    - Gros surcout cpu.
# 3) Division des boucles <=> 1) ?
#    - Surcoup taille programme
#    - +DEBIT+CPU / -PROG
# 4) Deroulage/Renroulage
#    - On deroule tout, on ajoute les transferts mem, on réenroule
#    - +DEBIT+CPU / -PROG

# EXEMPLE ALGO 3
#
# FOR int j = 0; j < 64; j++      # access L1 = 64*64
#  | FOR int i = 0; i < 64; i++   # access L0 = 64
#  |  | REF tab0[j][i]
#
# ===> EXEMPLES #DMA = 32 (#DMA < 1L0)
#
# FOR int j = 0; j < 64; j++
#  | FOR int mm = 0; mm < 2; mm++
#  |  | DMA tab[l][mm*32] -> BUFF # 32
#  |  | FOR int i2 = 0; i2 < 32; i2++
#  |  |  | REF BUFF[i2]
#
# ===> EXEMPLES #DMA = 128 (#DMA > 2L0 && #DMA < 1L1)
#
# FOR int j = 0; j < 64; j++
#  | DMA tab[j][0] -> BUFF # 128         # INSERT HERE
#  | FOR int mm = 0; mm < 2; mm++, j++   #
#  |  | FOR int i = 0; i < 64; i++
#  |  |  | REF BUFF[i + mm*64]
#
# ===> EXEMPLES #DMA = 64*8 (#DMA > 8L0 && #DMA < 1L1)
#
# FOR int j = 0; j < 64; j++
#  | DMA tab[j][0] -> BUFF #DMA          # INSERT HERE
#  | FOR int mm = 0; mm < 8; mm++, j++   #
#  |  | FOR int i = 0; i < 64; i++
#  |  |  | REF BUFF[i + mm*64]
#


# EXEMPLE ALGO 3 UNALIGNED
# FOR int j = 0; j < 6; j++      # access L1 = 6*6
#  | FOR int i = 0; i < 6; i++   # access L0 = 6
#  |  | REF tab0[j][i]
#
# ===> EXEMPLES #DMA = 25 (#DMA > 4 L0 && #DMA < 1L1)
# ===> VERSION AVEC IF
# FOR int j = 0; j < 6; j++
#  | DMA tab[j][0] -> BUFF #MIN(4*6, (6-j)*4)) # INSERT HERE
#  | FOR int mm = 0; mm < 4 && j < 6; mm++, j++   #
#  |  | FOR int i = 0; i < 6; i++
#  |  |  | REF BUFF[i + mm*6]
#
# ===> VERSION AVEC DUPLICATION BOUCLE
# int j = 0
# FOR ; j < NEW; j++      # NEW = LAST - LAST%4
#  | DMA tab[j][0] -> BUFF #(4*6)) # INSERT HERE
#  | FOR int mm = 0; mm < 4; mm++, j++
#  |  | FOR int i = 0; i < 6; i++
#  |  |  | REF BUFF[i + mm*6]
# DMA tab[j][0] -> BUFF #(2*4)
# FOR int mm = 0; mm < 2; mm ++
#  | FOR int i = 0; i < 6; i++
#  |  | REF BUFF[i + mm*6]
#
# # + Aucun curcout cpu !!
# # - Programme plus gros / plus compliqué à unroll ?


def main(filename):
    ast = at.file_to_ast(filename)
    do_memory_mapping(ast)
    print("CGEN")
    print(at.ast_to_c_highlight(ast))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
