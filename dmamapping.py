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


import sys


def do_memory_mapping(ast):
    for topfor in at.c_ast_get_all_topfor(ast):
        log.debug("TOP FORS:")
        # print(at.ast_to_c_highlight(topfor))
        refs = at.c_ast_get_all_top_ref(topfor)
        nb_refs = len(refs)
        log.debug(f"TOP REFS ({nb_refs}):")
        for i, ref in enumerate((refs)):
            log.debug(f"{at.ast_to_c(ref):20} RW={at.c_ast_ref_is_rw(ast, ref)}")
            dma_mapping_algo3(ast, ref, i)


def dma_mapping_algo3(ast, ref, iref):
    """ """
    # Analyse reference
    (
        for_nodes,
        ref_name,
        ref_access_names,
        loops_access_names,
        loops_access_l,
        loops_access_l_cum,
        ref_is_read,
        ref_is_write,
    ) = at.c_ast_ref_analyse(ast, ref)

    if 'input' in ref_name:
        return
    if 'weights' in ref_name:
        return
    
    # Remove __SMA__
    # for i, name in reversed(list(enumerate(loops_access_names))):
    #     if "__SMA__" in name:    
    #         loops_access_names.pop(i)
    #         loops_access_l.pop(i)

    loops_ref_access_l = [
        v if n in ref_access_names else 1
        for n, v in zip(loops_access_names, loops_access_l)
    ]
    loops_ref_access_l_cum = list(np.cumprod(loops_ref_access_l))
    ref_l = list(filter(lambda x: x!=1, loops_ref_access_l)) # TODO use def !
    ref_l_cum = list(np.cumprod(ref_l))
    loops_access_l_cum = None

    log.debug(f'========== DMA MAPPING {at.ast_to_c(ref)}')
    log.debug(f"{loops_access_names=}")
    log.debug(f"{loops_access_l=}")
    log.debug(f"{ref_name=}")
    log.debug(f"{ref_access_names=}")
    log.debug(f"{ref_l=}")
    log.debug(f"{ref_l_cum=}")
    log.debug(f"{loops_ref_access_l=}")
    log.debug(f"{loops_ref_access_l_cum=}")
    log.debug(f"{ref_is_read=}")
    log.debug(f"{ref_is_write=}")

    # Not a cube
    if ref_access_names != loops_access_names:
        # is lower cube ?
        def contain_ordered(listin, data):
            if listin == []:
                return True
            if data == []:
                return False
            if listin[0] == data[0]:
                return contain_ordered(listin[1:], data[1:])
            else:
                return contain_ordered(listin, data[1:])

        if not contain_ordered(ref_access_names, loops_access_names):
            raise Exception(
                f"Invalid memory mapping: {ref_access_names} != {loops_access_names} (TODO)"
            )
            # TODO: /!\ Lecture ne correspondant pas aux index direct exemple tab[i+1] !
            # TODO Partitionnement mémoire. Exemple Toeplitz matrix.
            # https://www.rle.mit.edu/eems/wp-content/uploads/2019/06/Tutorial-on-DNN-04-Kernel-Computations.pdf
            # Slide 25

    # Find where insert DMA LD/ST
    IL = 0
    if DMA_SIZE >= loops_ref_access_l_cum[-1]:  # We only need to do 1 transfert
        IL = -1
    else:
        while DMA_SIZE > loops_ref_access_l_cum[IL]:
            IL += 1


    # Find how to insert DMA LD/ST
    IR = 0
    if DMA_SIZE >= ref_l_cum[-1]:  # We only need to do 1 transfert
        IR = -1
    else:
        while DMA_SIZE > ref_l_cum[IR]:
            IR += 1

    log.debug(f"{IL=}")
    log.debug(f"{IR=}")

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
    elif IL == 0:
        raise Exception('Unimplemented')
    else:
        topcomp = for_nodes[IL].stmt
    
    if IR == -1: # Array < DMA
        log.debug('--- WRAP MODE')
        # Compute memory mapping
        inds = (0 for i, name in enumerate(ref_access_names))
        tab_rw = ref_name + "".join(reversed(list((f"[{index}]" for index in inds))))
        log.debug(f"substitute {(tab_rw)} # mapped @ {buffer_name}")
        # Insert transactions
        topcomp.block_items.insert(
            0, stmt_c_to_ast(f"int {size_name} = {loops_ref_access_l_cum[-1]};")
        )
        topcomp.block_items.insert(1, stmt_c_to_ast(f'void * {adr_name} = {"&" + tab_rw};'))
        if ref_is_read:  # insert LD
            topcomp.block_items.insert(2, expr_c_to_ast(Gencode.cgen_dma_ld(*cgen_dma_args)))
        if ref_is_write:  # Insert ST
            topcomp.block_items.append(expr_c_to_ast(Gencode.cgen_dma_st(*cgen_dma_args)))
        dma_efficiency = loops_ref_access_l_cum[-1] / DMA_SIZE
        # Update ref
        buff_adr = "+".join(
            f"{i}*{cumprod}"
            for i, cumprod in zip(
                ref_access_names,
                chain((1,), ref_l_cum[0:-1]),
            )
        )
        ast_buff = expr_c_to_ast(f"{buffer_name}[{buff_adr}]")
        ref.name = ast_buff.name  # Copy node
        ref.subscript = ast_buff.subscript
        ref.name.name = buffer_name  # Only to change name ID name

    else:
        if IR == 0: # Divise
            nb_repeat = ref_l[0] / DMA_SIZE
            nb_repeat_int = ceil(nb_repeat)
            dma_transfer_size = DMA_SIZE
            nb_residual_int = ref_l[0] % DMA_SIZE
        else: # Repeat
            nb_repeat = DMA_SIZE / loops_ref_access_l_cum[IL - 1]
            nb_repeat_int = floor(nb_repeat)
            dma_transfer_size = nb_repeat_int * loops_ref_access_l_cum[IL - 1]
            nb_residual_int = loops_access_l[IL - 1] % nb_repeat_int
        
        dma_efficiency = dma_transfer_size / DMA_SIZE
        log.debug('--- ' + ('DIVISE' if IR == 0 else 'REPEAT') + ' MODE')
        log.debug(f"{nb_repeat=}, {nb_repeat_int=}, {nb_residual_int=}")
        log.debug(f"{dma_transfer_size=}/{DMA_SIZE=} = {dma_transfer_size/DMA_SIZE}")

        # Find the for @ IL
        
        ast_sub_for = c_ast.Compound(topcomp.block_items)
        # print(at.ast_to_c_highlight(ast_sub_for))

        # print(ast_to_c_highlight(ast_sub_for))

        # replace tab <-> BUFF
        # TODO outdated (/!\ smaller cube)
        if IR == 0: # Divise
            buff_adr = iter_name
        else: # Repeat
            buff_adr = "+".join(
                chain(
                    (
                        f"{i}*{cumprod}"
                        for i, cumprod in zip(
                            ref_access_names[0:IR],
                            chain((1,), ref_l_cum),
                        )
                    ),
                    (f"{iter_name}*{ref_l_cum[IR-1]}",),
                )
            )
        ast_buff = expr_c_to_ast(f"{buffer_name}[{buff_adr}]")
        ref.name = ast_buff.name  # Copy node
        ref.subscript = ast_buff.subscript

        if IR == 0: # Divise
            inds = (name if i >= IL - 1 else 0 for i, name in enumerate(ref_access_names))
            # TODO
        else: # Repeat
            inds = (name if i > IL - 1 else 0 for i, name in enumerate(ref_access_names))
        
        tab_rw = ref_name + "".join(reversed(list((f"[{index}]" for index in inds))))
        log.debug(f"substitute {(tab_rw)} # mapped @ {buffer_name}s")
        body_repeat = DMA_SIZE if IR == 0 else nb_repeat_int

            
        stmts = []
        stmts.append(stmt_c_to_ast(f"static int {iter_name};"))
        stmts.append(stmt_c_to_ast(f"static int {size_name};"))
        stmts.append(stmt_c_to_ast(f'static void * {adr_name};'))

        
        if IR == 0: # Divise
            if nb_residual_int:
                size = f"MIN({dma_transfer_size}, ({loops_access_l[IL]}-{loops_access_names[IL]}))"
            else:
                size = str(DMA_SIZE)
        else: # Repeat
            size = f"MIN({dma_transfer_size}, ({loops_access_l[IL]}-{loops_access_names[IL]})*{loops_ref_access_l_cum[IL-1]})"


        stmts.append(stmt_c_to_ast(f"if ({current_name} % {body_repeat} == 0) {{{iter_name} = 0; {size_name} = {size}; {adr_name} = {'&' + tab_rw};}}"))
        if ref_is_read:
            stmts.append(stmt_c_to_ast(f"if ({current_name} % {body_repeat} == 0) {{{Gencode.cgen_dma_ld(*cgen_dma_args)};}}"))
        
        for stmt in ast_sub_for.block_items:
            stmts.append(stmt)

        stmts.append(stmt_c_to_ast(f'{iter_name}++;'))

        if ref_is_write:
            stmts.append(stmt_c_to_ast(f"if ({iter_name} == {size_name}) {{{Gencode.cgen_dma_st(*cgen_dma_args)};}}"))


        # print(ast_to_c_highlight(ast_intermediate))
        topcomp.block_items = stmts
    
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
