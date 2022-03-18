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

DMA_SIZE = 129
from math import ceil, floor


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
        print("TOP FORS:")
        print(at.ast_to_c_highlight(topfor))
        refs = at.c_ast_get_all_top_ref(topfor)
        nb_refs = len(refs)
        print(f"TOP REFS ({nb_refs}):")
        for i, ref in enumerate((refs)):
            print(f"{at.ast_to_c(ref):20} RW={at.c_ast_ref_is_rw(ast, ref)}")
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

    loops_ref_access_l = [
        v if n in ref_access_names else 1
        for n, v in zip(loops_access_names, loops_access_l)
    ]
    loops_ref_access_l_cum = list(np.cumprod(loops_ref_access_l))

    print("for_nodes=", list(map(type, for_nodes)))
    print(f"{loops_access_names=}")
    print(f"{loops_access_l=}")
    print(f"{loops_access_l_cum=}")

    print(f"{ref_name=}")
    print(f"{ref_access_names=}")
    print(f"{loops_ref_access_l=}")
    print(f"{loops_ref_access_l_cum=}")

    print(f"{ref_is_read=}")
    print(f"{ref_is_write=}")

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

    print(f"{IL=}")

    buffer_name = f"__SMA__dma{iref}"
    adr_name = f"__SMA__adr{iref}"
    size_name = f"__SMA__size{iref}"
    iter_name = f"__SMA__i{iref}"
    cgen_dma_args = (adr_name, buffer_name, size_name)

    if IL == -1:
        # Compute memory mapping
        inds = (0 for i, name in enumerate(ref_access_names))
        tab_rw = ref_name + "".join(
            reversed(list((f"[{index}]" for index in inds)))
        )
        print(f"substitute {(tab_rw)} # mapped @ {buffer_name}s")
        # Insert transactions
        top_for = for_nodes[-1]
        content = at.c_ast_get_upper_node(ast, top_for).block_items
        content.insert(
            0, stmt_c_to_ast(f"int {size_name} = {loops_ref_access_l_cum[-1]};")
        )
        content.insert(1, stmt_c_to_ast(f'void * {adr_name} = {"&" + tab_rw};'))
        if ref_is_read:  # insert LD
            content.insert(2, expr_c_to_ast(Gencode.cgen_dma_ld(*cgen_dma_args)))
        if ref_is_write:  # Insert ST
            content.append(expr_c_to_ast(Gencode.cgen_dma_st(*cgen_dma_args)))
        dma_efficiency = loops_ref_access_l_cum[-1] / DMA_SIZE
        # Update ref
        ref.name.name = buffer_name  # Only to change name ID name

    elif IL == 0:
        # Divise elementary loop
        raise Exception("Unimplemented TODO")
        pass
    else:
        # Repeat multiple time the elementary loop
        nb_repeat = DMA_SIZE / loops_access_l_cum[IL - 1]
        nb_repeat_int = floor(nb_repeat)
        dma_transfer_size = nb_repeat_int * loops_access_l_cum[IL - 1]
        nb_residual_int = loops_access_l[IL - 1] % nb_repeat_int
        dma_efficiency = dma_transfer_size / DMA_SIZE
        print(f"{nb_repeat=}, {nb_repeat_int=}, {nb_residual_int=}")
        print(f"{dma_transfer_size=}/{DMA_SIZE=} = {dma_transfer_size/DMA_SIZE}")

        # Find the for @ IL
        inner_top_for = for_nodes[IL]
        ast_sub_for = inner_top_for.stmt  # compound.pop(position)
        # print(ast_to_c_highlight(ast_sub_for))

        # replace tab <-> BUFF
        # TODO outdated (/!\ smaller cube)
        buff_adr = "+".join(
            chain(
                (
                    f"{i}*{cumprod}"
                    for i, cumprod in zip(
                        ref_access_names[0:IL],
                        chain((1,), loops_access_l_cum[0:IL]),
                    )
                ),
                (f"{iter_name}*{loops_access_l_cum[IL-1]}",),
            )
        )
        ast_buff = expr_c_to_ast(f"{buffer_name}[{buff_adr}]")

        ref.name = ast_buff.name  # Copy node
        ref.subscript = ast_buff.subscript
        inds = (
            name if i > IL - 1 else 0 for i, name in enumerate(ref_access_names)
        )
        tab_rw = ref_name + "".join(
            reversed(list((f"[{index}]" for index in inds)))
        )
        print(f"substitute {(tab_rw)} # mapped @ {buffer_name}s")
        # print(ast_to_c_highlight(ast_sub_for))

        stmts = []
        stmts.append(stmt_c_to_ast(f'void * {adr_name} = {"&" + tab_rw};'))
        stmts.append(
            stmt_c_to_ast(
                f"int {size_name} = MIN({dma_transfer_size}, ({loops_access_l[IL]}-{loops_access_names[IL]})*{nb_repeat_int}*{loops_access_l_cum[IL-1]});"
            )
        )
        if ref_is_read:
            stmts.append(stmt_c_to_ast(f"{Gencode.cgen_dma_ld(*cgen_dma_args)};"))

        if nb_residual_int:
            body = c_ast.Compound(
                [
                    c_ast.If(
                        expr_c_to_ast(
                            f"{loops_access_names[IL]} < {loops_access_l[IL]}"
                        ),
                        c_ast.Compound(
                            [
                                ast_sub_for,
                                expr_c_to_ast(f"{loops_access_names[IL]}++"),
                            ]
                        ),
                        None,
                    )
                ]
            )
        else:
            body = ast_sub_for

        stmts.append(
            c_ast.For(
                expr_c_to_ast(f"int {iter_name} = {0}"),
                expr_c_to_ast(f"{iter_name} < {nb_repeat_int}"),
                expr_c_to_ast(f"{iter_name}++"),
                body,
            )
        )
        if ref_is_write:
            stmts.append(stmt_c_to_ast(f"{Gencode.cgen_dma_st(*cgen_dma_args)};"))
        stmts.append(stmt_c_to_ast(f"{loops_access_names[IL]}--;"))  # TODO Beark

        ast_intermediate = c_ast.Compound(stmts)
        # print(ast_to_c_highlight(ast_intermediate))
        inner_top_for.stmt = ast_intermediate

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
