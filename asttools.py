
from pycparser import parse_file, c_generator, CParser
from pycparser import c_ast
from pycparser import plyparser

from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import TerminalFormatter

import numpy as np

def file_to_ast(file, use_cpp=True):
    return parse_file(file, use_cpp=True)

def c_to_ast(code):
    try:
        ast = CParser().parse(code)
        return ast
    except plyparser.ParseError:
        print("Error Gencode; invalid code:")
        print(c_highlight(code))
        raise

def compound_c_to_ast(code):
    try:
        ast = CParser().parse("void azertytreza(){{" + code + "}}")
        return ast.ext[0].body.block_items[0]
    except plyparser.ParseError:
        print("Error Gencode; invalid code:")
        print(c_highlight(code))
        raise


def stmt_c_to_ast(code):
    res = compound_c_to_ast(f"{code}")
    if len(res.block_items) != 1:
        print("Error Gencode; invalid code:")
        print(c_highlight(code))
        print(res)
        raise
    return res.block_items[0]


def expr_c_to_ast(code):
    res = compound_c_to_ast(f"{code};")
    if len(res.block_items) != 1:
        print("Error Gencode; invalid code:")
        print(c_highlight(code))
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


def c_highlight(code):
    return highlight(code, CLexer(), TerminalFormatter(bg="dark", linenos=True))


def c_ast_For_extract_l(node):
    """Return the for Bounds"""
    # /!\ Very restrictive
    try:
        var_loop_name = node.init.decls[0].name
        assert node.init.decls[0].init.value == "0"
        assert node.cond.op == "<"
        assert node.cond.left.name == var_loop_name
        l = node.cond.right.value
        return (var_loop_name, int("0"), int(l))
    except:
        print("Invalid for:")
        print(ast_to_c_highlight(node))
        raise


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
    ref_is_read, ref_is_write = c_ast_ref_is_rw(ast, ref)
    return (
        for_nodes,
        ref_name,
        ref_access_names,
        loops_access_names,
        loops_access_l,
        loops_access_l_cum,
        ref_is_read,
        ref_is_write,
    )


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
            self.is_read = not node.op == "="  # else # <= , >=, +=, ...
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

