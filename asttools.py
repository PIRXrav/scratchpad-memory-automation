from pycparser import parse_file, c_generator, CParser
from pycparser import c_ast
from pycparser import plyparser

from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import TerminalFormatter

from itertools import chain
from more_itertools import one
from copy import deepcopy

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
    return c_highlight(ast_to_c(ast))


def c_highlight(code):
    formatter = TerminalFormatter(bg="dark", linenos=True)
    return highlight(code, CLexer(), formatter)


def c_ast_for_get_l(node):
    """Return the for Bounds"""
    # /!\ Very restrictive
    try:
        var_loop_name = node.init.decls[0].name
        assert len(node.init.decls) == 1
        assert node.cond.op == "<"
        assert node.cond.left.name == var_loop_name
        assert node.next.op == "p++"
        assert node.next.expr.name == var_loop_name
        return (var_loop_name, node.init.decls[0].init, node.cond.right)
    except Exception:
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


def c_ast_constant_to_int(node):
    if node.type == "int":
        return int(node.value)
    else:
        raise Exception(f"Invalid node {node}")


def c_ast_arraydecl_get_all(decl):
    """
    Analyse the arraydecl
    """

    class ArrayDeclVisitor(c_ast.NodeVisitor):
        """
        Decl(name='input',
            quals=[
                ],
            align=[
                ],
            storage=[
                    ],
            funcspec=[
                    ],
            type=ArrayDecl(type=TypeDecl(declname='input',
                                        quals=[
                                                ],
                                        align=None,
                                        type=IdentifierType(names=['char'
                                                                    ]
                                                            )
                                        ),
                            dim=Constant(type='int',
                                        value='16'
                                        ),
                            dim_quals=[
                                    ]
                            ),
            init=None,
            bitsize=None
            ))
        """

        def __init__(self):
            self.dims = []
            self.name = None
            self.type = None

        def generic_visit(self, node):
            raise Exception("Did not visit a decl", node)

        def visit_Decl(self, node):
            self.name = node.name
            self.visit(node.type)

        def visit_ArrayDecl(self, node):
            self.dims.append(node.dim)
            self.visit(node.type)

        def visit_TypeDecl(self, node):
            assert self.name == node.declname
            self.type = node

    v = ArrayDeclVisitor()
    v.visit(decl)
    assert v.name is not None
    assert v.type is not None
    return v.name, v.type, v.dims


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
            self.res.append(node.subscript)
            self.visit(node.name)

        def visit_ID(self, node):
            self.name = node.name

    rv = RefVisitor()
    rv.visit(ref)
    return rv.name, rv.res


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


def c_ast_delc_to_ptr_decl(decl):
    """Convert a decl to a pointer of the same type"""
    if decl.type.__class__ == c_ast.ArrayDecl:
        decl.type = c_ast.PtrDecl([], deepcopy(decl.type.type))
    else:
        raise Exception(f"Unimplemented type: {decl.type}")
    return decl


def fun_get_name(fun):
    return fun.decl.name


def fun_set_name(fun, name):
    fun.decl.type.type.declname = name
    fun.decl.name = name


def c_ast_ref_update(ref, name, subscript):
    ref.name = name
    ref.subscript = subscript


def c_ast_get_all_id(ast):
    """Return all idendifiers ID(name='')"""
    class IDVisitor(c_ast.NodeVisitor):
        def __init__(self):
            self.res = []

        def visit_ID(self, node):
            self.res.append(node)

    nv = IDVisitor()
    nv.visit(ast)
    return nv.res

def c_ast_get_all_id_by_name(ast, name):
    """Return all idendifiers filtered by name ID(name='')"""
    return filter(lambda node: node.name == name, c_ast_get_all_id(ast))


def c_ast_for_update_name(fornode, name):
    """
    For(init=DeclList(decls=[Decl(name='x',
                              quals=[
                                    ],
                              align=[
                                    ],
                              storage=[
                                      ],
                              funcspec=[
                                       ],
                              type=TypeDecl(declname='x',
                                            quals=[
                                                  ],
                                            align=None,
                                            type=IdentifierType(names=['int'
                                                                      ]
                                                                )
                                            ),
                              init=Constant(type='int',
                                            value='0'
                                            ),
                              bitsize=None
                              )
                        ]
                  ),
    cond=BinaryOp(op='<',
                  left=ID(name='x'
                          ),
                  right=BinaryOp(op='+',
                                 left=BinaryOp(op='-',
                                               left=Constant(type='int',
                                                             value='256'
                                                             ),
                                               right=Constant(type='int',
                                                              value='3'
                                                              )
                                               ),
                                 right=Constant(type='int',
                                                value='1'
                                                )
                                 )
                  ),
    next=UnaryOp(op='p++',
                 expr=ID(name='x'
                         )
                 ),
    """
    fornode.init.decls[0].name = name
    fornode.init.decls[0].type.declname = name
    for idnode in chain(*map(c_ast_get_all_id, [fornode.cond, fornode.next])):
        idnode.name = name


def c_ast_for_update_l(fornode, astl):
    """
    cond=BinaryOp(op='<',
                  left=ID(name='x'
                          ),
                  right=BinaryOp(op='+',
                                 left=BinaryOp(op='-',
                                               left=Constant(type='int',
                                                             value='256'
                                                             ),
                                               right=Constant(type='int',
                                                              value='3'
                                                              )
                                               ),
                                 right=Constant(type='int',
                                                value='1'
                                                )
                                 )
    """
    fornode.cond.right = astl


def test_c_ast_get_all_id():
    ast = stmt_c_to_ast('for(int n = 0; n < 10; n++) {n = x + n; x += x * 2;}')
    idnodes = list(c_ast_get_all_id_by_name(ast, 'n'))
    assert len(idnodes) == 4


def c_ast_replace_id(ast, name, new_ast):
    """Replace ids"""
    class ReplaceVisitor(c_ast.NodeVisitor):
        def __init__(self, name, new_ast):
            self.res = []
            self.name = name
            self.new_ast = new_ast

        def generic_visit(self, node):
            self.res.append(node)
            for c in node:  #
                if self.visit(c):
                    # lets hack slots !
                    slot = one(filter(lambda s: getattr(node, s) == c, node.__slots__))
                    # Update the corresponding slot with new_ast
                    setattr(node, slot, deepcopy(new_ast))  # copy is preferable

            return False

        def visit_ID(self, node):
            return node.name == self.name

    nv = ReplaceVisitor(name, new_ast)
    nv.visit(ast)
    return ast


def test_c_ast_replace_id():
    ast = stmt_c_to_ast('for(int n = 0; n < 10; n++) {n = x + n; x += x * 2;}')
    new_ast = expr_c_to_ast("r * pi")
    c_ast_replace_id(ast, 'x', new_ast)
    # print(ast)
    assert len(list(c_ast_get_all_id_by_name(ast, 'r'))) == 3  # catch new r
    # print(ast_to_c_highlight(ast))

def c_ast_typename(type_s):
    """
    Typename(name=None,
                      quals=[
                            ],
                      align=None,
                      type=PtrDecl(quals=[
                                         ],
                                   type=TypeDecl(declname=None,
                                                 quals=[
                                                       ],
                                                 align=None,
                                                 type=IdentifierType(names=['int'
                                                                           ]
                                                                     )
                                                 )
                                   )
                      ),
    """
    nbptr_type = type_s.count('*')
    base_type_s = type_s.replace('*', '')
    base_type = c_ast.TypeDecl(None, [], None, c_ast.IdentifierType([base_type_s]))
    for _ in range(nbptr_type):
        base_type = c_ast.PtrDecl([], base_type)
    return c_ast.Typename(None, [], None, base_type)

def c_ast_get_parent(ast, nodes):
    class ParentVisitor(c_ast.NodeVisitor):
        def __init__(self, nodes):
            self.res = []
            self.nodes = set(nodes)

        def generic_visit(self, node):
            if node in self.nodes:
                return True
            for c in node:
                if self.visit(c):
                    # lets hack slots !
                    slot = one(filter(lambda s: getattr(node, s) == c, node.__slots__))
                    self.res.append((node, slot, c))  # $node.$slot = $c
            return False

    nv = ParentVisitor(nodes)
    nv.visit(ast)
    return nv.res

def c_ast_cast_nodes(ast, nodes, cast_type):
    """Replace node by (cast_type)node"""
    for parent, slot, node in c_ast_get_parent(ast, nodes):
        setattr(parent, slot, c_ast.Cast(c_ast_typename(cast_type), node))
    return ast


def c_ast_update_ref_dereference_type(ast, nodes, type_decl):
    """Replace node by (cast_type)node"""
    if type(type_decl) is not c_ast.TypeDecl:
        raise Exception(f"{type_decl} is not of type c_ast.TypeDecl")
    for parent, slot, node in c_ast_get_parent(ast, nodes):
        assert getattr(parent, slot) is node
        to_type = c_ast.Typename(None, [], None, c_ast.PtrDecl([], type_decl))
        new_node = c_ast.UnaryOp('&', node)
        new_node = c_ast.Cast(to_type, new_node)
        new_node = c_ast.UnaryOp('*', new_node)
        setattr(parent, slot, new_node)
    return ast

# def c_ast_update_ref_dereference_type(ast, nodes, deref_type):
#     """Replace node by (cast_type)node"""
#     for parent, slot, node in c_ast_get_parent(ast, nodes):
#         node = c_ast.UnaryOp('&', node)
#         node = c_ast.Cast(c_ast_typename(deref_type + '*'), node)
#         node = c_ast.UnaryOp('*', node)
#         setattr(parent, slot, node)
#     return ast

def test_c_ast_cast_node():
    # print((expr_c_to_ast('56 + *(int*)&x')))
    ast = stmt_c_to_ast('for(int n = 0; n < 10; n++) {n = x + n; x += x * 2;}')
    xids = list(c_ast_get_all_id_by_name(ast, 'x'))
    c_ast_cast_nodes(ast, xids, 'mytype**')


def test_c_ast_update_ref_dereference_type():
    ast = stmt_c_to_ast('for(int n = 0; n < 10; n++) {n = x + n; x += x * 2;}')
    xids = list(c_ast_get_all_id_by_name(ast, 'x'))
    type_decl = c_ast.TypeDecl(None, [], None, c_ast.IdentifierType(['__i64']))
    print(type_decl)
    c_ast_update_ref_dereference_type(ast, xids, type_decl)
    print(ast_to_c_highlight(ast))


if __name__ == '__main__':
    test_c_ast_get_all_id()
    test_c_ast_replace_id()
    test_c_ast_cast_node()
    test_c_ast_update_ref_dereference_type()
