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


DMA_SIZE = 129
from math import ceil, floor

from pycparser import plyparser

class Gencode:
    @classmethod
    def cgen_dma_ld(self, adr, buff, size):
        return f"DMA_LD({adr}, {buff}, {size})"

    @classmethod
    def cgen_dma_st(self, adr, buff, size):
        return f"DMA_ST({adr}, {buff}, {size})"

import sys


def compound_c_to_ast(code):
    try:
        ast = CParser().parse("void azertytreza(){{" + code + "}}")
        return ast.ext[0].body.block_items[0]
    except plyparser.ParseError:
        print("Error Gencode; invalid code:")
        print(c_highight(code))
        raise


def stmt_c_to_ast(code):
    res = compound_c_to_ast(f"{code}")
    if len(res.block_items) != 1:
        print("Error Gencode; invalid code:")
        print(c_highight(code))
        print(res)
        raise
    return res.block_items[0]

def expr_c_to_ast(code):
    res = compound_c_to_ast(f"{code};")
    if len(res.block_items) != 1:
        print("Error Gencode; invalid code:")
        print(c_highight(code))
        print(res)
        raise
    return res.block_items[0]

class DmaBufferHandler:
    def __init__(self):
        pass

    def get_dma_buffer(self):
        pass


def ast_to_c(ast):
    return c_generator.CGenerator().visit(ast)


def ast_to_c_highlight(ast):
    return highlight(
        ast_to_c(ast), CLexer(), TerminalFormatter(bg="dark", linenos=True)
    )


def c_highight(code):
    return highlight(code, CLexer(), TerminalFormatter(bg="dark", linenos=True))


def c_ast_For_extract_l(node):
    """Return the for Bounds"""
    var_loop_name = node.init.decls[0].name
    # /!\ Very restrictive
    assert node.init.decls[0].init.value == "0"
    assert node.cond.op == "<"
    assert node.cond.left.name == var_loop_name
    l = node.cond.right.value
    return (var_loop_name, int("0"), int(l))


def c_ast_get_for_fathers(ast, node):
    class ForFathersVisitor(c_ast.NodeVisitor):
        def __init__(self, node):
            self.node = node
            self.forfathers = None
            self.stack = []

        def visit_For(self, node):
            self.stack.append(node)
            for n in node:
                self.visit(n)
            self.stack.pop()

        def generic_visit(self, node):
            if node == self.node:
                self.forfathers = list(reversed(self.stack))
            else:
                for n in node:
                    self.visit(n)
        
    nv = ForFathersVisitor(node)
    nv.visit(ast)
    return nv.forfathers

def c_ast_get_all_top_ref(node):
    class AllRefVisitor(c_ast.NodeVisitor):
        def __init__(self):
            self.refs = []

        def visit_ArrayRef(self, node):
            self.refs.append(node)
        
    nv = AllRefVisitor()
    nv.visit(node)
    return nv.refs

def c_ast_get_upper_node(ast, node):
    class UpperNodeVisitor(c_ast.NodeVisitor):
        def __init__(self, node):
            self.node = node
            self.uppernode = None
            self.compoundnode = None

        def visit_Compound(self, node):
            self.compoundnode = node
            for n in node:
                self.visit(n)

        def generic_visit(self, node):
            if node == self.node:
                self.uppernode = self.compoundnode
            for n in node:
                self.visit(n)
    
    unv = UpperNodeVisitor(node)
    unv.visit(ast)
    uppernode = unv.uppernode
    return uppernode

def c_ast_ref_get_l(ref):
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

    rv = RefVisitor()
    rv.visit(ref)
    return rv.name, rv.res

def c_ast_ref_analyse(ast, ref):
    # Compute fathers list for nodes
    for_nodes = c_ast_get_for_fathers(ast, ref)
    # Comute L value for all for nodes
    l_tree = list(map(c_ast_For_extract_l, for_nodes))
    loops_access_l = [l[2] for l in l_tree]
    loops_access_l_cum = list(np.cumprod(loops_access_l))
    loops_access_names = [l[0] for l in l_tree]

    ref_name, ref_access_names = c_ast_ref_get_l(ref)
    ref_is_read, ref_is_write  = c_ast_ref_is_rw(ast, ref)
    return for_nodes, ref_name, ref_access_names, loops_access_names, loops_access_l, loops_access_l_cum, ref_is_read, ref_is_write

def c_ast_get_all_topfor(ast):
    class AllTopForVisitor(c_ast.NodeVisitor):
        def __init__(self):
            self.fors = []

        def visit_For(self, node):
            self.fors.append(node)
            
    nv = AllTopForVisitor()
    nv.visit(ast)
    return nv.fors


def c_ast_ref_is_rw(ast, ref):
    """
    Test is_read, is_write to ArrayRef Node
    """
    class RefRWVisitor(c_ast.NodeVisitor):
        def __init__(self, node):
            # Default is READ only
            self.is_write = False
            self.is_read = True
            self.node = node
            self.res = None

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
            if node == self.node:
                self.res = (self.is_read, self.is_write)

    nv = RefRWVisitor(ref)
    nv.visit(ast)
    return nv.res
    # # Check decoration
    # for ref in c_ast_get_all_ref(ast):
    #     try:
    #         if ref.is_read == None:
    #             raise
    #         if ref.is_write == None:
    #             raise
    #     except:
    #         Exception("Error occured during decorate_all_ref_rw")
    # Impossible do decorate tree: __slots__ class :/

class AstToolkit:
    def __init__(self, filename):
        # Compute pycparser ast
        self.ast = parse_file(filename, use_cpp=True)
   
    def do_memory_mapping(self):
        # print(self.mast)
        ast = self.ast
        for topfor in c_ast_get_all_topfor(ast):
            print("TOP FORS:")
            print(ast_to_c_highlight(topfor))
            refs = c_ast_get_all_top_ref(topfor)
            print("TOP REFS:")
            for ref in refs:
                print(f'{ast_to_c(ref):20} RW={c_ast_ref_is_rw(ast, ref)}')
            
            for ref in [refs[0]]:
                self.dma_mapping_algo3(ref)

    def exportc(self):
        generator = c_generator.CGenerator()
        return generator.visit(self.ast)


    def dma_mapping_algo3(self, ref):
        """ """
        ast = self.ast
        # Analyse reference
        (
            for_nodes,
            ref_name,
            ref_access_names,
            loops_access_names,
            loops_access_l,
            loops_access_l_cum,
            ref_is_read,
            ref_is_write
        ) = c_ast_ref_analyse(ast, ref)

    
        loops_ref_access_l = [v if n in ref_access_names else 1 
                             for n, v in zip(loops_access_names, loops_access_l)]
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
                print(listin, data)
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
        if DMA_SIZE >= loops_ref_access_l_cum[-1]: # We only need to do 1 transfert
            IL = -1
        else:
            while DMA_SIZE > loops_ref_access_l_cum[IL]:
                IL += 1

        print(f"{IL=}")

        buffer_name = "__SMA__dma0"
        adr_name = '__SMA__adr0'
        size_name = '__SMA__size0'
        cgen_dma_args = (adr_name, buffer_name, size_name)

        if IL == -1:
            # Compute memory mapping
            inds = (0 for i, name in enumerate(ref_access_names))
            tab_rw = ref_name + "".join(reversed(list((f"[{index}]" for index in inds))))
            print(f"substitute {(tab_rw)} # mapped @ {buffer_name}s")
            # Insert transactions
            top_for = for_nodes[-1]
            content = c_ast_get_upper_node(ast, top_for).block_items
            content.insert(0, stmt_c_to_ast(f'int {size_name} = {loops_ref_access_l_cum[-1]};'))
            content.insert(1, stmt_c_to_ast(f'void * {adr_name} = {"&" + tab_rw};'))           
            if ref_is_read: # insert LD
                content.insert(2, expr_c_to_ast(Gencode.cgen_dma_ld(*cgen_dma_args)))
            if ref_is_write: # Insert ST
                content.append(expr_c_to_ast(Gencode.cgen_dma_st(*cgen_dma_args)))
            dma_efficiency = loops_ref_access_l_cum[-1]/DMA_SIZE
            # Update ref
            ref.name.name = buffer_name # Only to change name ID name
    
            
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
            ast_sub_for = inner_top_for.stmt # compound.pop(position)
            # print(ast_to_c_highlight(ast_sub_for))

            # replace tab <-> BUFF
            # TODO outdated
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
            ast_buff = expr_c_to_ast(f"{buffer_name}[{buff_adr}]")
            # print(ast_to_c_highlight(ast_buff))
            ref.name = ast_buff.name  # Copy node
            ref.subscript = ast_buff.subscript
            inds = (name if i > IL - 1 else 0 for i, name in enumerate(ref_access_names))
            tab_rw = ref_name + "".join(reversed(list((f"[{index}]" for index in inds))))
            # print(f"substitute ref->{(buff_rw)} mapped @ {(tab_rw)}")
            # print(ast_to_c_highlight(ast_sub_for))
            
            stmts=[]
            stmts.append(stmt_c_to_ast(f'void * {adr_name} = {"&" + tab_rw};'))
            stmts.append(stmt_c_to_ast(f'int {size_name} = MIN({dma_transfer_size}, ({loops_access_l[IL]}-{loops_access_names[IL]})*{nb_repeat_int}*{loops_access_l_cum[IL-1]});'))
            if ref_is_read:
                stmts.append(stmt_c_to_ast(f'{Gencode.cgen_dma_ld(*cgen_dma_args)};'))
            stmts.append(c_ast.For(expr_c_to_ast(f'int mm = {0}'),
                             expr_c_to_ast(f'mm < {nb_repeat_int} && {loops_access_names[IL]} < {loops_access_l[IL]}'),
                             expr_c_to_ast(f'mm++, {loops_access_names[IL]}++'),
                             ast_sub_for))
            if ref_is_write:
                stmts.append(stmt_c_to_ast(f'{Gencode.cgen_dma_st(*cgen_dma_args)};'))
            stmts.append(stmt_c_to_ast(f'{loops_access_names[IL]}--;')) # TODO Beark

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
    ast = AstToolkit(filename)
    ast.do_memory_mapping()
    print("CGEN")
    print(ast_to_c_highlight(ast.ast))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
