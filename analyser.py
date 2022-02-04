"""
SCRATCHPAD MEMORY AUTOMATION
"""

import sys
from pycparser import parse_file, c_generator

from pycparser import c_ast

from collections import defaultdict


def ast_to_c(ast):
    return c_generator.CGenerator().visit(ast)


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


class MyNode:

    rd = " | "

    def render(self):
        pass

    def __str__(self):
        return self.render(0)


class MyFileAST(MyNode):
    def __init__(self, node):
        self.c_ast_node = node
        self.inner = []

    def render(self, dept):
        res = self.rd * dept + "FILE\n"
        for mnode in self.inner:
            res += mnode.render(dept + 1)
        return res


class MyFuncDef(MyNode):
    def __init__(self, node):
        self.c_ast_node = node
        self.inner = []

    def render(self, dept):
        res = self.rd * dept + "FUNC " + ast_to_c(self.c_ast_node.decl) + "\n"
        for mnode in self.inner:
            res += mnode.render(dept + 1)
        return res


class MyFor(MyNode):
    def __init__(self, node):
        self.c_ast_node = node
        # TODO check bonded
        self.inner = []

    def render(self, dept):
        def str_For(node):
            return "; ".join(map(ast_to_c, (node.init, node.cond, node.next)))

        res = self.rd * dept + "FOR " + str_For(self.c_ast_node) + "\n"
        for mnode in self.inner:
            res += mnode.render(dept + 1)
        return res


class MyArrayRef(MyNode):
    def __init__(self, node):
        self.c_ast_node = node
        # TODO check array constant/i acces
        self.varname = ""
        self.access_patern = None

    def render(self, dept):
        return self.rd * dept + "REF " + ast_to_c(self.c_ast_node) + "\n"


class AstToolkit:
    def __init__(self, filename):
        # Compute pycparser ast
        self.ast = parse_file(filename, use_cpp=True)
        # Compute our ast
        mv = MyVisitor()
        mv.visit(self.ast)
        self.mast = mv.stack[-1]

    def test(self):
        print(self.mast)

    def exportc(self):
        generator = c_generator.CGenerator()
        return generator.visit(self.ast)


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


def main(filename):
    ast = AstToolkit(filename)
    ast.test()
    print("CGEN")
    print(ast.exportc())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
