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


from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import TerminalFormatter

import numpy as np


def ast_to_c(ast):
    return c_generator.CGenerator().visit(ast)


def ast_to_c_highlight(ast):
    return highlight(
        ast_to_c(ast), CLexer(), TerminalFormatter(bg="dark", linenos=True)
    )


def c_highight(code):
    return highlight(code, CLexer(), TerminalFormatter(bg="dark", linenos=True))


class MyVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.stack = []

    def visit_For(self, node):
        res = MyFor(node)

        self.stack.append(res.inner)
        self.visit(node.stmt)
        self.stack.pop()

        self.stack[-1].append(res)

    def visit_FuncDef(self, node):
        # decl, param_decs, body
        res = MyFuncDef(node)

        self.stack.append(res.inner)
        self.visit(node.body)
        self.stack.pop()

        self.stack[-1].append(res)

    def visit_ArrayRef(self, node):
        # name, subscript
        res = MyArrayRef(node)
        self.stack[-1].append(res)

    def visit_FileAST(self, node):
        # ext
        res = MyFileAST(node)

        self.stack.append(res.inner)
        self.visit(node.ext)
        self.stack.pop()

        self.stack.append(res)


from itertools import chain


class MyNode:

    rd = " | "

    def __init__(self, node):
        self.inner = []
        self.c_ast_node = node
        self.father = None

    def update_fathers(self):
        for node in self.inner:
            node.father = self
            node.update_fathers()

    def get_all_mrefs(self):
        return (
            [self]
            if type(self) is MyArrayRef
            else chain(*(mnode.get_all_mrefs() for mnode in self.inner))
        )

    def iter_father(self):
        mnode = self
        while not mnode.father is None:
            yield mnode
            mnode = mnode.father

    def render(self):
        pass

    def __str__(self):
        return self.render(0)


class MyFileAST(MyNode):
    def __init__(self, node):
        super().__init__(node)

    def render(self, dept):
        res = self.rd * dept + "FILE\n"
        for mnode in self.inner:
            res += mnode.render(dept + 1)
        return res


class MyFuncDef(MyNode):
    def __init__(self, node):
        super().__init__(node)

    def render(self, dept):
        res = self.rd * dept + "FUNC " + ast_to_c(self.c_ast_node.decl) + "\n"
        for mnode in self.inner:
            res += mnode.render(dept + 1)
        return res


class MyFor(MyNode):
    def __init__(self, node):
        super().__init__(node)
        # TODO check bonded

    def render(self, dept):
        def str_For(node):
            return "; ".join(map(ast_to_c, (node.init, node.cond, node.next)))

        res = self.rd * dept + "FOR " + str_For(self.c_ast_node) + "\n"
        for mnode in self.inner:
            res += mnode.render(dept + 1)
        return res

    def extract_l(self):
        """Return the for Bounds"""
        # print(self.c_ast_node)
        var_loop_name = self.c_ast_node.init.decls[0].name
        # /!\ Very restrictive
        assert self.c_ast_node.init.decls[0].init.value == "0"
        assert self.c_ast_node.cond.op == "<"
        assert self.c_ast_node.cond.left.name == var_loop_name
        l = self.c_ast_node.cond.right.value
        return (var_loop_name, int("0"), int(l))


class MyArrayRef(MyNode):
    def __init__(self, node):
        super().__init__(node)
        # TODO check array constant/i acces
        self.is_write = None
        self.is_read = None

    def render(self, dept):
        return self.rd * dept + "REF " + ast_to_c(self.c_ast_node) + "\n"

    def extract_l_tree(self):
        """
        l_tree format : [(i_name, 0, N), (i_name2, 0, M), ...]
        """
        return (
            mfor_node.extract_l()
            for mfor_node in filter(lambda x: type(x) is MyFor, self.iter_father())
        )

    


    def analyse(self):
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
                self.res = []
                self.name = None

            def generic_visit(self, node):
                raise Exception("Did not visit a ref", node)

            def visit_ArrayRef(self, node):
                self.res.append(node.subscript.name)  # TODO i+k ...
                self.visit(node.name)

            def visit_ID(self, node):
                self.name = node.name

        # Compute fathers list for nodes
        for_nodes = list(filter(lambda x: type(x) is MyFor, self.iter_father()))
        # Comute L value for all for nodes
        l_tree = list(self.extract_l_tree())

        loops_access_l = [l[2] for l in l_tree]
        loops_access_l_cum = list(np.cumprod(loops_access_l))
        loops_access_names = [l[0] for l in l_tree]

        rv = RefVisitor()
        rv.visit(self.c_ast_node)
        ref_name, ref_access_names = rv.name, rv.res

        if ref_access_names != loops_access_names:
            raise Exception(
                f"Invalid memory mapping: {ref_access_names} != {loops_access_names} (TODO)"
            )
            # TODO: /!\ Lecture ne correspondant pas aux index direct exemple tab[i+1] !
            # TODO Partitionnement mémoire. Exemple Toeplitz matrix.
            # https://www.rle.mit.edu/eems/wp-content/uploads/2019/06/Tutorial-on-DNN-04-Kernel-Computations.pdf
            # Slide 25

        return for_nodes, ref_name, ref_access_names, loops_access_l, loops_access_l_cum


class AstToolkit:
    def __init__(self, filename):
        # Compute pycparser ast
        self.ast = parse_file(filename, use_cpp=True)
        # Compute our ast
        mv = MyVisitor()
        mv.visit(self.ast)
        self.mast = mv.stack[-1]
        self.mast.update_fathers()
        for ref in self.mast.get_all_mrefs():  # TODO remove, use visitor
            print(ref)
        self.decorate_mrefs_rw()

    def decorate_mrefs_rw(self):
        """
        Add is_read, is_write to RefNode
        """
        class RefRWVisitor(c_ast.NodeVisitor):
            """
          
            """
            def __init__(self, mrefs):
                self.mrefs = mrefs
                # Default is READ only
                self.is_write = False
                self.is_read = True

            def visit_Assignment(self, node):
                # Check L value
                self.is_write = True
                self.is_read = not node.op == '=' # else # <= , >=, +=, ...
                self.visit(node.lvalue)
                
                # Check R value
                self.is_write = False
                self.is_read = True
                self.visit(node.rvalue)
                
                # Default is READ only
                self.is_write = False
                self.is_read = True

            def visit_ArrayRef(self, node):
                for mref in self.mrefs:
                    if mref.c_ast_node == node:
                        mref.is_read = self.is_read
                        mref.is_write = self.is_write
                        return
                raise Exception("Unknown node", node)


        rrwv = RefRWVisitor(list(self.mast.get_all_mrefs()))
        rrwv.visit(self.ast)
        # Check decoration
        for mref in self.mast.get_all_mrefs():
            if mref.is_read == None:
                raise Exception("Error occured during decorate_mrefs_rw")
            if mref.is_write == None:
                raise Exception("Error occured during decorate_mrefs_rw")
                

    def do_memory_mapping(self):
        print(self.mast)
        for mref in self.mast.get_all_mrefs():
            dma_mapping_algo3(mref)

    def exportc(self):
        generator = c_generator.CGenerator()
        return generator.visit(self.ast)


DMA_SIZE = 129
from math import ceil, floor

from pycparser import plyparser


class Gencode:
    @classmethod
    def cgen_dma_ld(self, adr, buff, size):
        return f"DMA_LD({adr}, {buff}, {size});"

    @classmethod
    def cgen_dma_st(self, adr, buff, size):
        return f"DMA_ST({adr}, {buff}, {size});"

import sys


def compound_c_to_ast(code):
    try:
        ast = CParser().parse("void azertytreza(){{" + code + "}}")
        return ast.ext[0].body.block_items[0]
    except plyparser.ParseError:
        print("Error Gencode; invalid code:")
        print(c_highight(code))
        raise


def expr_c_to_ast(code):
    return compound_c_to_ast(f"{code};").block_items[0]

class DmaBufferHandler:
    def __init__(self):
        pass

    def get_dma_buffer(self):
        pass

def dma_mapping_algo3(mref):
    """ """
    # Analyse reference
    (
        for_nodes,
        ref_name,
        ref_access_names,
        loops_access_l,
        loops_access_l_cum,
    ) = mref.analyse()
    print(f"{ref_name=}")
    print(f"{ref_access_names=}")
    print(f"{loops_access_l=}")
    print(f"{loops_access_l_cum=}")
    print(f"{mref.is_read=}")
    print(f"{mref.is_write=}")


    # Find where insert DMA LD/ST
    IL = 0
    while DMA_SIZE > loops_access_l_cum[IL]:
        IL += 1

    print(f"{IL=}")

    if IL == 0:
        # Divise elementary loop
        raise Exception("Unimplemented TODO")
        pass
    else:
        # Repeat multiple time the elementaty loop
        nb_repeat = DMA_SIZE / loops_access_l_cum[IL - 1]
        nb_repeat_int = floor(nb_repeat)
        dma_transfer_size = nb_repeat_int * loops_access_l_cum[IL - 1]
        nb_residual_int = loops_access_l[IL - 1] % nb_repeat_int
        dma_efficiency = dma_transfer_size / DMA_SIZE
        print(f"{nb_repeat=}, {nb_repeat_int=}, {nb_residual_int=}")
        print(f"{dma_transfer_size=}/{DMA_SIZE=} = {dma_transfer_size/DMA_SIZE}")

        # Find the for @ IL
        inner_top_for = for_nodes[IL].c_ast_node
        ast_sub_for = inner_top_for.stmt # compound.pop(position)
        # print(ast_to_c_highlight(ast_sub_for))

        # replace tab <-> BUFF
        buffer_name = "__SMA__dma0"
        buff_adr = "+".join(
            chain(
                (
                    f"{i}*{cumprod}"
                    for i, cumprod in zip(
                        ref_access_names[0:IL],
                        chain((1,), loops_access_l_cum[0:IL]),
                    )
                ),
                (f"mm*{loops_access_l_cum[IL-1]}",),
            )
        )
        buff_rw = f"{buffer_name}[{buff_adr}]"
        ast_buff = expr_c_to_ast(buff_rw)
        # print(ast_to_c_highlight(ast_buff))
        mref.c_ast_node.name = ast_buff.name  # Copy node
        mref.c_ast_node.subscript = ast_buff.subscript
        inds = (name if i > IL - 1 else 0 for i, name in enumerate(ref_access_names))
        tab_rw = ref_name + "".join(reversed(list((f"[{index}]" for index in inds))))
        # print(f"substitute ref->{(buff_rw)} mapped @ {(tab_rw)}")
        # print(ast_to_c_highlight(ast_sub_for))

        # ALGO 3; Note the {ast_to_c(ast_sub_for)}
        adr_name = '__SMA__adr0'
        size_name = '__SMA__size0'
        cgen_dma_args = (adr_name, buffer_name, size_name)
        algo_c = (''
            + f'void * {adr_name} = {"&" + tab_rw};\n'
            + f'int {size_name} = MIN({dma_transfer_size}, ({loops_access_l[IL]}-{ref_access_names[IL]})*{nb_repeat_int}*{loops_access_l_cum[IL-1]});\n'
            + (Gencode.cgen_dma_ld(*cgen_dma_args) if mref.is_read is True else '')
            + f"for(int mm = 0; mm < {nb_repeat_int} && {ref_access_names[IL]} < {loops_access_l[IL]}; mm++, {ref_access_names[IL]}++){{{ast_to_c(ast_sub_for)}}}\n"
            + (Gencode.cgen_dma_st(*cgen_dma_args) if mref.is_write is True else '')
            + f'{ref_access_names[IL]}--;' # TODO Beark
        )
        ast_intermediate = compound_c_to_ast(algo_c)
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
    ast = AstToolkit(filename)
    ast.do_memory_mapping()
    print("CGEN")
    print(ast_to_c_highlight(ast.ast))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
