"""
SCRATCHPAD MEMORY AUTOMATION
"""

from operator import index
import sys
from pycparser import parse_file, c_generator, CParser

from pycparser import c_ast

from collections import defaultdict


from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import TerminalFormatter

def ast_to_c(ast):
    return c_generator.CGenerator().visit(ast)

def ast_to_c_highlight(ast):
    return highlight(ast_to_c(ast), CLexer(), TerminalFormatter(bg='dark', linenos=True))


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
    
    def get_all_refs(self):
        return [self] if type(self) is MyArrayRef else \
               chain(*(mnode.get_all_refs() for mnode in self.inner))

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
        """ Return the for Bounds """
        # print(self.c_ast_node)
        var_loop_name = self.c_ast_node.init.decls[0].name
        # /!\ Very restrictive
        assert self.c_ast_node.init.decls[0].init.value == '0'
        assert self.c_ast_node.cond.op == '<'
        assert self.c_ast_node.cond.left.name == var_loop_name
        l = self.c_ast_node.cond.right.value
        return (var_loop_name, int('0'), int(l))


class MyArrayRef(MyNode):
    def __init__(self, node):
        super().__init__(node)
        # TODO check array constant/i acces
        self.varname = ""
        self.access_patern = None

    def render(self, dept):
        return self.rd * dept + "REF " + ast_to_c(self.c_ast_node) + "\n"

    def extract_l_tree(self):
        return (mfor_node.extract_l() for mfor_node in 
                filter(lambda x: type(x) is MyFor, self.iter_father()))

class AstToolkit:
    def __init__(self, filename):
        # Compute pycparser ast
        self.ast = parse_file(filename, use_cpp=True)
        # Compute our ast
        mv = MyVisitor()
        mv.visit(self.ast)
        self.mast = mv.stack[-1]
        self.mast.update_fathers()
        for ref in self.mast.get_all_refs():
            print(ref)
    
        
    def test(self):
        print(self.mast)
        for mref in self.mast.get_all_refs():
            dma_mapping_algo3(mref)

    def exportc(self):
        generator = c_generator.CGenerator()
        return generator.visit(self.ast)



DMA_SIZE =  129
from math import ceil, floor


def dma_mapping_algo3(mref):
    """
    l_tree format : [(name, 0, N), (name2, 0, M), ...]
    """

    for_nodes = list(filter(lambda x: type(x) is MyFor, mref.iter_father()))
    l_tree = list(mref.extract_l_tree())

    print(for_nodes, l_tree)

    # Compute cumulative acces
    arr_names = [l[0] for l in l_tree]
    arr_l = [l[2] for l in l_tree]
    # BEARK ! :/    
    arr_l_cum = [None for _ in arr_l]
    arr_l_cum[0] = arr_l[0]
    for i in range(1, len(arr_l)):
        arr_l_cum[i] = arr_l_cum[i-1] * arr_l[i]

    print(f'{arr_l=} {arr_l_cum=}')

    index_loop = 0
    while DMA_SIZE > arr_l_cum[index_loop]:
        index_loop+=1
    
    print(f'{index_loop=}')

    if index_loop == 0:
        # Divise elementary loop
        pass
    else:
        # Repeat multiple time the elementaty loop
        nb_repeat = DMA_SIZE / arr_l_cum[index_loop-1]
        nb_repeat_int = floor(nb_repeat)
        dma_transfer_size = nb_repeat_int * arr_l_cum[index_loop-1]
        nb_residual_int =arr_l[index_loop-1] % nb_repeat_int
        dma_efficiency = dma_transfer_size/DMA_SIZE
        print(f'{nb_repeat=}, {nb_repeat_int=}, {nb_residual_int=}')
        print(f'{dma_transfer_size=}/{DMA_SIZE=} = {dma_transfer_size/DMA_SIZE}')
        
        # Find the for @ index_loop
        compound = for_nodes[index_loop].c_ast_node.stmt.block_items
        position = compound.index(for_nodes[index_loop-1].c_ast_node)
        ast_sub_for = compound.pop(position)
        # TODO replace tab -> BUFF
        print(ast_to_c_highlight(ast_sub_for))

        # ALGO 3; Note the {ast_to_c(ast_sub_for)}
        algo_c = f'DMA_READ({"TODO"}[{"TODO"}], BUFF, MIN({dma_transfer_size}, ({arr_l[index_loop]}-{arr_names[index_loop]})*{nb_repeat_int}));\
                  for(int mm = 0; mm < {nb_repeat_int} && {arr_names[index_loop]} < {arr_l[index_loop]}; mm++, {arr_names[index_loop]}++){{{ast_to_c(ast_sub_for)}}}'
        ast_intermediate = CParser().parse('void azertytreza(){{' + algo_c + '}}').ext[0].body.block_items[0]
        print(ast_to_c_highlight(ast_intermediate))


        compound.insert(position, ast_intermediate)
       

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
# FOR int j = 0; j < 6; j++
#  | DMA tab[j][0] -> BUFF #MIN(4*6, (6-j)*4)) # INSERT HERE
#  | FOR int mm = 0; mm < 4 && j < 6; mm++, j++   #
#  |  | FOR int i = 0; i < 6; i++
#  |  |  | REF BUFF[i + mm*6]
#

# TODO: /!\ Lecture ne correspondant pas aux index direct exemple tab[i+1] !

def main(filename):
    ast = AstToolkit(filename)
    ast.test()
    print("CGEN")
    print(ast_to_c_highlight(ast.ast))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
