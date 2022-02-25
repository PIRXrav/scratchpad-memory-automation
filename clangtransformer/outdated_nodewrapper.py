

import sys

def _repr(obj):
    """
    Get the representation of an object, with dedicated pprint-like format for lists.
    """
    if isinstance(obj, list):
        return '[' + (',\n '.join((_repr(e).replace('\n', '\n ') for e in obj))) + '\n]'
    else:
        return repr(obj)

class Node(object):
    __slots__ = ()
    """ Abstract base class for AST nodes.
    """
    def __repr__(self):
        """ Generates a python representation of the current node
        """
        result = self.__class__.__name__ + '('

        indent = ''
        separator = ''
        for name in self.__slots__[:-2]:
            result += separator
            result += indent
            result += name + '=' + (_repr(getattr(self, name)).replace('\n', '\n  ' + (' ' * (len(name) + len(self.__class__.__name__)))))

            separator = ','
            indent = '\n ' + (' ' * len(self.__class__.__name__))

        result += indent + ')'

        return result

    def children(self):
        """ A sequence of all children that are Nodes
        """
        pass

    def show(self, buf=sys.stdout, offset=0, attrnames=False, nodenames=False, showcoord=False, _my_node_name=None):
        """ Pretty print the Node and all its attributes and
            children (recursively) to a buffer.

            buf:
                Open IO buffer into which the Node is printed.

            offset:
                Initial offset (amount of leading spaces)

            attrnames:
                True if you want to see the attribute names in
                name=value pairs. False to only see the values.

            nodenames:
                True if you want to see the actual node names
                within their parents.

            showcoord:
                Do you want the coordinates of each Node to be
                displayed.
        """
        lead = ' ' * offset
        if nodenames and _my_node_name is not None:
            buf.write(lead + self.__class__.__name__+ ' <' + _my_node_name + '>: ')
        else:
            buf.write(lead + self.__class__.__name__+ ': ')

        if self.attr_names:
            if attrnames:
                nvlist = [(n, getattr(self,n)) for n in self.attr_names]
                attrstr = ', '.join('%s=%s' % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ', '.join('%s' % v for v in vlist)
            buf.write(attrstr)

        if showcoord:
            buf.write(' (at %s)' % self.coord)
        buf.write('\n')

        for (child_name, child) in self.children():
            child.show(
                buf,
                offset=offset + 2,
                attrnames=attrnames,
                nodenames=nodenames,
                showcoord=showcoord,
                _my_node_name=child_name)


class NodeVisitor(object):
    """ A base NodeVisitor class for visiting c_ast nodes.
        Subclass it and define your own visit_XXX methods, where
        XXX is the class name you want to visit with these
        methods.

        For example:

        class ConstantVisitor(NodeVisitor):
            def __init__(self):
                self.values = []

            def visit_Constant(self, node):
                self.values.append(node.value)

        Creates a list of values of all the constant nodes
        encountered below the given node. To use it:

        cv = ConstantVisitor()
        cv.visit(node)

        Notes:

        *   generic_visit() will be called for AST nodes for which
            no visit_XXX method was defined.
        *   The children of nodes for which a visit_XXX was
            defined will not be visited - if you need this, call
            generic_visit() on the node.
            You can use:
                NodeVisitor.generic_visit(self, node)
        *   Modeled after Python's own AST visiting facilities
            (the ast module of Python 3.0)
    """

    _method_cache = None

    def visit(self, node):
        """ Visit a node.
        """

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__, None)
        if visitor is None:
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node)

    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a
            node. Implements preorder visiting of the node.
        """
        for c in node:
            self.visit(c)



# TODO ?
class ArrayDecl(Node):
    __slots__ = ('type', 'dim', 'dim_quals', 'coord', '__weakref__')
    def __init__(self, type, dim, dim_quals, coord=None):
        self.type = type
        self.dim = dim
        self.dim_quals = dim_quals
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.dim is not None: nodelist.append(("dim", self.dim))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.dim is not None:
            yield self.dim

    attr_names = ('dim_quals', )

# [C99 6.5.2.1] Array Subscripting.
class Node_ARRAY_SUBSCRIPT_EXPR(Node):
    __slots__ = ('name', 'subscript', 'coord', '__weakref__')
    def __init__(self, name, subscript, coord=None):
        self.name = name
        self.subscript = subscript
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.subscript is not None: nodelist.append(("subscript", self.subscript))
        return tuple(nodelist)

    def __iter__(self):
        if self.name is not None:
            yield self.name
        if self.subscript is not None:
            yield self.subscript

    attr_names = ()

# A builtin binary operation expression such as "x + y" or
# "x <= y".
class Node_BINARY_OPERATOR(Node):
    __slots__ = ('op', 'left', 'right', 'coord', '__weakref__')
    def __init__(self, op, left, right, coord=None):
        self.op = op
        self.left = left
        self.right = right
        self.coord = coord

    def children(self):
        nodelist = []
        if self.left is not None: nodelist.append(("left", self.left))
        if self.right is not None: nodelist.append(("right", self.right))
        return tuple(nodelist)

    def __iter__(self):
        if self.left is not None:
            yield self.left
        if self.right is not None:
            yield self.right

    attr_names = ('op', )

# A break statement.
class Node_BREAK_STMT(Node):
    __slots__ = ('coord', '__weakref__')
    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()


# A case statement.
class Node_CASE_STMT(Node):
    __slots__ = ('expr', 'stmts', 'coord', '__weakref__')
    def __init__(self, expr, stmts, coord=None):
        self.expr = expr
        self.stmts = stmts
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        for i, child in enumerate(self.stmts or []):
            nodelist.append(("stmts[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr
        for child in (self.stmts or []):
            yield child

    attr_names = ()


# An explicit cast in C (C99 6.5.4) or a C-style cast in C++
# (C++ [expr.cast]), which uses the syntax (Type)expr.
#
class Node_CSTYLE_CAST_EXPR(Node):
    __slots__ = ('to_type', 'expr', 'coord', '__weakref__')
    def __init__(self, to_type, expr, coord=None):
        self.to_type = to_type
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.to_type is not None: nodelist.append(("to_type", self.to_type))
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    def __iter__(self):
        if self.to_type is not None:
            yield self.to_type
        if self.expr is not None:
            yield self.expr

    attr_names = ()

# A compound statement
class Node_COMPOUND_STMT(Node):
    __slots__ = ('block_items', 'coord', '__weakref__')
    def __init__(self, block_items, coord=None):
        self.block_items = block_items
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.block_items or []):
            nodelist.append(("block_items[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.block_items or []):
            yield child

    attr_names = ()


# [C99 6.5.2.5] #DODO Not sure !
class Node_COMPOUND_LITERAL_EXPR(Node):
    __slots__ = ('type', 'init', 'coord', '__weakref__')
    def __init__(self, type, init, coord=None):
        self.type = type
        self.init = init
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.init is not None: nodelist.append(("init", self.init))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.init is not None:
            yield self.init

    attr_names = ()

# # TODO ?
# class Constant(Node):
#     __slots__ = ('type', 'value', 'coord', '__weakref__')
#     def __init__(self, type, value, coord=None):
#         self.type = type
#         self.value = value
#         self.coord = coord

#     def children(self):
#         nodelist = []
#         return tuple(nodelist)

#     def __iter__(self):
#         return
#         yield

#     attr_names = ('type', 'value', )


# A continue statement.
class Node_CONTINUE_STMT(Node):
    __slots__ = ('coord', '__weakref__')
    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()

class Decl(Node):
    __slots__ = ('name', 'quals', 'align', 'storage', 'funcspec', 'type', 'init', 'bitsize', 'coord', '__weakref__')
    def __init__(self, name, quals, align, storage, funcspec, type, init, bitsize, coord=None):
        self.name = name
        self.quals = quals
        self.align = align
        self.storage = storage
        self.funcspec = funcspec
        self.type = type
        self.init = init
        self.bitsize = bitsize
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.init is not None: nodelist.append(("init", self.init))
        if self.bitsize is not None: nodelist.append(("bitsize", self.bitsize))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.init is not None:
            yield self.init
        if self.bitsize is not None:
            yield self.bitsize

    attr_names = ('name', 'quals', 'align', 'storage', 'funcspec', )

class DeclList(Node):
    __slots__ = ('decls', 'coord', '__weakref__')
    def __init__(self, decls, coord=None):
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.decls or []):
            yield child

    attr_names = ()

class Default(Node):
    __slots__ = ('stmts', 'coord', '__weakref__')
    def __init__(self, stmts, coord=None):
        self.stmts = stmts
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.stmts or []):
            nodelist.append(("stmts[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.stmts or []):
            yield child

    attr_names = ()

class DoWhile(Node):
    __slots__ = ('cond', 'stmt', 'coord', '__weakref__')
    def __init__(self, cond, stmt, coord=None):
        self.cond = cond
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.stmt is not None:
            yield self.stmt

    attr_names = ()

class EllipsisParam(Node):
    __slots__ = ('coord', '__weakref__')
    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()

class EmptyStatement(Node):
    __slots__ = ('coord', '__weakref__')
    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()

class Enum(Node):
    __slots__ = ('name', 'values', 'coord', '__weakref__')
    def __init__(self, name, values, coord=None):
        self.name = name
        self.values = values
        self.coord = coord

    def children(self):
        nodelist = []
        if self.values is not None: nodelist.append(("values", self.values))
        return tuple(nodelist)

    def __iter__(self):
        if self.values is not None:
            yield self.values

    attr_names = ('name', )

class Enumerator(Node):
    __slots__ = ('name', 'value', 'coord', '__weakref__')
    def __init__(self, name, value, coord=None):
        self.name = name
        self.value = value
        self.coord = coord

    def children(self):
        nodelist = []
        if self.value is not None: nodelist.append(("value", self.value))
        return tuple(nodelist)

    def __iter__(self):
        if self.value is not None:
            yield self.value

    attr_names = ('name', )

class EnumeratorList(Node):
    __slots__ = ('enumerators', 'coord', '__weakref__')
    def __init__(self, enumerators, coord=None):
        self.enumerators = enumerators
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.enumerators or []):
            nodelist.append(("enumerators[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.enumerators or []):
            yield child

    attr_names = ()

class ExprList(Node):
    __slots__ = ('exprs', 'coord', '__weakref__')
    def __init__(self, exprs, coord=None):
        self.exprs = exprs
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(("exprs[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.exprs or []):
            yield child

    attr_names = ()


# Cursor that represents the translation unit itself.
#
# The translation unit cursor exists primarily to act as the root cursor for
# traversing the contents of a translation unit.
class Node_TRANSLATION_UNIT(Node):
    __slots__ = ('ext', 'coord', '__weakref__')
    def __init__(self, ext, coord=None):
        self.ext = ext
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.ext or []):
            nodelist.append(("ext[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.ext or []):
            yield child

    attr_names = ()

# A for statement.
class Node_FOR_STMT(Node):
    __slots__ = ('init', 'cond', 'next', 'stmt', 'coord', '__weakref__')
    def __init__(self, init, cond, next, stmt, coord=None):
        self.init = init
        self.cond = cond
        self.next = next
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.init is not None: nodelist.append(("init", self.init))
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.next is not None: nodelist.append(("next", self.next))
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.init is not None:
            yield self.init
        if self.cond is not None:
            yield self.cond
        if self.next is not None:
            yield self.next
        if self.stmt is not None:
            yield self.stmt

    attr_names = ()


# An expression that calls a function.
class Node_CALL_EXPR(Node):
    __slots__ = ('name', 'args', 'coord', '__weakref__')
    def __init__(self, name, args, coord=None):
        self.name = name
        self.args = args
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.args is not None: nodelist.append(("args", self.args))
        return tuple(nodelist)

    def __iter__(self):
        if self.name is not None:
            yield self.name
        if self.args is not None:
            yield self.args

    attr_names = ()


# class Node_FUNCTION_DECL(Node):
#     __slots__ = ('args', 'type', 'coord', '__weakref__')
#     def __init__(self, args, type, coord=None):
#         self.args = args
#         self.type = type
#         self.coord = coord

#     def children(self):
#         nodelist = []
#         if self.args is not None: nodelist.append(("args", self.args))
#         if self.type is not None: nodelist.append(("type", self.type))
#         return tuple(nodelist)

#     def __iter__(self):
#         if self.args is not None:
#             yield self.args
#         if self.type is not None:
#             yield self.type

#     attr_names = ()

# A function.
class Node_FUNCTION_DECL(Node):
    __slots__ = ('decl', 'param_decls', 'body', 'coord', '__weakref__')
    def __init__(self, decl, param_decls, body, coord=None):
        self.decl = decl
        self.param_decls = param_decls
        self.body = body
        self.coord = coord

    def children(self):
        nodelist = []
        if self.decl is not None: nodelist.append(("decl", self.decl))
        if self.body is not None: nodelist.append(("body", self.body))
        for i, child in enumerate(self.param_decls or []):
            nodelist.append(("param_decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        if self.decl is not None:
            yield self.decl
        if self.body is not None:
            yield self.body
        for child in (self.param_decls or []):
            yield child

    attr_names = ()

# A goto statement.
class Node_GOTO_STMT(Node):
    __slots__ = ('name', 'coord', '__weakref__')
    def __init__(self, name, coord=None):
        self.name = name
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    def __iter__(self):
        return
        yield

    attr_names = ('name', )

# TODO (No need ?) --DeclRefExpr
# class ID(Node):
#     __slots__ = ('name', 'coord', '__weakref__')
#     def __init__(self, name, coord=None):
#         self.name = name
#         self.coord = coord

#     def children(self):
#         nodelist = []
#         return tuple(nodelist)

#     def __iter__(self):
#         return
#         yield

#     attr_names = ('name', )

# TODO No need ? => -DeclRefExpr
# class IdentifierType(Node):
#     __slots__ = ('names', 'coord', '__weakref__')
#     def __init__(self, names, coord=None):
#         self.names = names
#         self.coord = coord

#     def children(self):
#         nodelist = []
#         return tuple(nodelist)

#     def __iter__(self):
#         return
#         yield

#     attr_names = ('names', )


# An if statement.
class Node_IF_STMT(Node):
    __slots__ = ('cond', 'iftrue', 'iffalse', 'coord', '__weakref__')
    def __init__(self, cond, iftrue, iffalse, coord=None):
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.iftrue is not None: nodelist.append(("iftrue", self.iftrue))
        if self.iffalse is not None: nodelist.append(("iffalse", self.iffalse))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.iftrue is not None:
            yield self.iftrue
        if self.iffalse is not None:
            yield self.iffalse

    attr_names = ()

# TODO ? 
# class InitList(Node):
#     __slots__ = ('exprs', 'coord', '__weakref__')
#     def __init__(self, exprs, coord=None):
#         self.exprs = exprs
#         self.coord = coord

#     def children(self):
#         nodelist = []
#         for i, child in enumerate(self.exprs or []):
#             nodelist.append(("exprs[%d]" % i, child))
#         return tuple(nodelist)

#     def __iter__(self):
#         for child in (self.exprs or []):
#             yield child

#     attr_names = ()


# A labelled statement in a function.
class Node_LABEL_STMT(Node):
    __slots__ = ('name', 'stmt', 'coord', '__weakref__')
    def __init__(self, name, stmt, coord=None):
        self.name = name
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.stmt is not None:
            yield self.stmt

    attr_names = ('name', )

# TODO ?
class NamedInitializer(Node):
    __slots__ = ('name', 'expr', 'coord', '__weakref__')
    def __init__(self, name, expr, coord=None):
        self.name = name
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        for i, child in enumerate(self.name or []):
            nodelist.append(("name[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr
        for child in (self.name or []):
            yield child

    attr_names = ()


# A function or method parameter.
class Node_PARM_DECL(Node):
    __slots__ = ('params', 'coord', '__weakref__')
    def __init__(self, params, coord=None):
        self.params = params
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.params or []):
            nodelist.append(("params[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.params or []):
            yield child

    attr_names = ()

# TODO ?
class PtrDecl(Node):
    __slots__ = ('quals', 'type', 'coord', '__weakref__')
    def __init__(self, quals, type, coord=None):
        self.quals = quals
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type

    attr_names = ('quals', )


# A return statement.
class Node_RETURN_STMT(Node):
    __slots__ = ('expr', 'coord', '__weakref__')
    def __init__(self, expr, coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr

    attr_names = ()

# A static_assert or _Static_assert Node
class Node_STATIC_ASSERT(Node):
    __slots__ = ('cond', 'message', 'coord', '__weakref__')
    def __init__(self, cond, message, coord=None):
        self.cond = cond
        self.message = message
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.message is not None: nodelist.append(("message", self.message))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.message is not None:
            yield self.message

    attr_names = ()

# A C or C++ struct.
class Node_STRUCT_DECL(Node):
    __slots__ = ('name', 'decls', 'coord', '__weakref__')
    def __init__(self, name, decls, coord=None):
        self.name = name
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.decls or []):
            yield child

    attr_names = ('name', )


# A field (in C) or non-static data member (in C++) in a struct, union, or C++
# class. TODO 
class Node_FIELD_DECL(Node):
    __slots__ = ('name', 'type', 'decls', 'coord', '__weakref__')
    def __init__(self, name, type, decls, coord=None):
        self.name = name
        self.type = type
        self.decls = decls
        self.coord = coord
    
    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.decls or []):
            yield child

# TODO ?
class StructRef(Node):
    __slots__ = ('name', 'type', 'field', 'coord', '__weakref__')
    def __init__(self, name, type, field, coord=None):
        self.name = name
        self.type = type
        self.field = field
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.field is not None: nodelist.append(("field", self.field))
        return tuple(nodelist)

    def __iter__(self):
        if self.name is not None:
            yield self.name
        if self.field is not None:
            yield self.field

    attr_names = ('type', )


# A switch statement.
class Node_SWITCH_STMT(Node):
    __slots__ = ('cond', 'stmt', 'coord', '__weakref__')
    def __init__(self, cond, stmt, coord=None):
        self.cond = cond
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.stmt is not None:
            yield self.stmt

    attr_names = ()


# The ? ternary operator.
class Node_CONDITIONAL_OPERATOR(Node):
    __slots__ = ('cond', 'iftrue', 'iffalse', 'coord', '__weakref__')
    def __init__(self, cond, iftrue, iffalse, coord=None):
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.iftrue is not None: nodelist.append(("iftrue", self.iftrue))
        if self.iffalse is not None: nodelist.append(("iffalse", self.iffalse))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.iftrue is not None:
            yield self.iftrue
        if self.iffalse is not None:
            yield self.iffalse

    attr_names = ()

class TypeDecl(Node):
    __slots__ = ('declname', 'quals', 'align', 'type', 'coord', '__weakref__')
    def __init__(self, declname, quals, align, type, coord=None):
        self.declname = declname
        self.quals = quals
        self.align = align
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type

    attr_names = ('declname', 'quals', 'align', )

class Typedef(Node):
    __slots__ = ('name', 'quals', 'storage', 'type', 'coord', '__weakref__')
    def __init__(self, name, quals, storage, type, coord=None):
        self.name = name
        self.quals = quals
        self.storage = storage
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type

    attr_names = ('name', 'quals', 'storage', )

class Typename(Node):
    __slots__ = ('name', 'quals', 'align', 'type', 'coord', '__weakref__')
    def __init__(self, name, quals, align, type, coord=None):
        self.name = name
        self.quals = quals
        self.align = align
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type

    attr_names = ('name', 'quals', 'align', )


# This represents the unary-expression's (except sizeof and
# alignof).
class Node_UNARY_OPERATOR(Node):
    __slots__ = ('op', 'expr', 'coord', '__weakref__')
    def __init__(self, op, expr, coord=None):
        self.op = op
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr

    attr_names = ('op', )

# A C or C++ union.
class Node_UNION_DECL(Node):
    __slots__ = ('name', 'decls', 'coord', '__weakref__')
    def __init__(self, name, decls, coord=None):
        self.name = name
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.decls or []):
            yield child

    attr_names = ('name', )

# A while statement.
class Node_WHILE_STMT(Node):
    __slots__ = ('cond', 'stmt', 'coord', '__weakref__')
    def __init__(self, cond, stmt, coord=None):
        self.cond = cond
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.stmt is not None:
            yield self.stmt

    attr_names = ()

class Pragma(Node):
    __slots__ = ('string', 'coord', '__weakref__')
    def __init__(self, string, coord=None):
        self.string = string
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    def __iter__(self):
        return
        yield

    attr_names = ('string', )


# A variable.
class Node_VAR_DECL(Node):
    __slots__ = ('name', 'type', '__weakref__')
    def __init__(self, name, type, coord=None):
        self.name = name
        self.type = type
        # self.coord=coord


# A reference to a type declaration.
#
# A type reference occurs anywhere where a type is named but not
# declared. 
#
# The typeclass is a declaration of size_type (CXCursor_TypeclassDecl),
# while the type of the variable "size" is referenced. The cursor
# referenced by the type of size is the typeclass for size_type.
class Node_TYPE_REF(Node):
    __slots__ = ('name', '__weakref__')
    def __init__(self, name, coord=None):
        self.name = name
        # self.coord = coord


# An attribute whoe specific kind is note exposed via this interface
class Node_UNEXPOSED_ATTR(Node):
    __slots__ = ('__weakref__')
    def __init__(self, coord=None):
        # self.coord = coord
        pass


###
# Declaration Kinds

# A declaration whose specific kind is not exposed via this interface.
#
# Unexposed declarations have the same operations as any other kind of
# declaration; one can extract their location information, spelling, find their
# classinitions, etc. However, the specific kind of the declaration is not
# reported.
class Node_UNEXPOSED_DECL(Node):
    def __init__(self):
        raise NotImplemented


# A C++ class.
class Node_CLASS_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An enumeration.
class Node_ENUM_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An enumerator constant.
class Node_ENUM_CONSTANT_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C @interface.
class Node_OBJC_INTERFACE_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C @interface for a category.
class Node_OBJC_CATEGORY_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C @protocol declaration.
class Node_OBJC_PROTOCOL_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C @property declaration.
class Node_OBJC_PROPERTY_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C instance variable.
class Node_OBJC_IVAR_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C instance method.
class Node_OBJC_INSTANCE_METHOD_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C class method.
class Node_OBJC_CLASS_METHOD_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C @implementation.
class Node_OBJC_IMPLEMENTATION_DECL(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C @implementation for a category.
class Node_OBJC_CATEGORY_IMPL_DECL(Node):
    def __init__(self):
        raise NotImplemented

# A typeclass.
class Node_TYPEclass_DECL(Node):
    def __init__(self):
        raise NotImplemented

# A C++ class method.
class Node_CXX_METHOD(Node):
    def __init__(self):
        raise NotImplemented

# A C++ namespace.
class Node_NAMESPACE(Node):
    def __init__(self):
        raise NotImplemented

# A linkage specification, e.g. 'extern "C"'.
class Node_LINKAGE_SPEC(Node):
    def __init__(self):
        raise NotImplemented

# A C++ constructor.
class Node_CONSTRUCTOR(Node):
    def __init__(self):
        raise NotImplemented

# A C++ destructor.
class Node_DESTRUCTOR(Node):
    def __init__(self):
        raise NotImplemented

# A C++ conversion function.
class Node_CONVERSION_FUNCTION(Node):
    def __init__(self):
        raise NotImplemented

# A C++ template type parameter
class Node_TEMPLATE_TYPE_PARAMETER(Node):
    def __init__(self):
        raise NotImplemented

# A C++ non-type template parameter.
class Node_TEMPLATE_NON_TYPE_PARAMETER(Node):
    def __init__(self):
        raise NotImplemented

# A C++ template template parameter.
class Node_TEMPLATE_TEMPLATE_PARAMETER(Node):
    def __init__(self):
        raise NotImplemented

# A C++ function template.
class Node_FUNCTION_TEMPLATE(Node):
    def __init__(self):
        raise NotImplemented

# A C++ class template.
class Node_CLASS_TEMPLATE(Node):
    def __init__(self):
        raise NotImplemented

# A C++ class template partial specialization.
class Node_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION(Node):
    def __init__(self):
        raise NotImplemented

# A C++ namespace alias declaration.
class Node_NAMESPACE_ALIAS(Node):
    def __init__(self):
        raise NotImplemented

# A C++ using directive
class Node_USING_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# A C++ using declaration
class Node_USING_DECLARATION(Node):
    def __init__(self):
        raise NotImplemented

# A Type alias decl.
class Node_TYPE_ALIAS_DECL(Node):
    def __init__(self):
        raise NotImplemented

# A Objective-C synthesize decl
class Node_OBJC_SYNTHESIZE_DECL(Node):
    def __init__(self):
        raise NotImplemented

# A Objective-C dynamic decl
class Node_OBJC_DYNAMIC_DECL(Node):
    def __init__(self):
        raise NotImplemented

# A C++ access specifier decl.
class Node_CXX_ACCESS_SPEC_DECL(Node):
    def __init__(self):
        raise NotImplemented


###
# Reference Kinds

class Node_OBJC_SUPER_CLASS_REF(Node):
    def __init__(self):
        raise NotImplemented
class Node_OBJC_PROTOCOL_REF(Node):
    def __init__(self):
        raise NotImplemented
class Node_OBJC_CLASS_REF(Node):
    def __init__(self):
        raise NotImplemented


class Node_CXX_BASE_SPECIFIER(Node):
    def __init__(self):
        raise NotImplemented

# A reference to a class template, function template, template
# template parameter, or class template partial specialization.
class Node_TEMPLATE_REF(Node):
    def __init__(self):
        raise NotImplemented

# A reference to a namespace or namepsace alias.
class Node_NAMESPACE_REF(Node):
    def __init__(self):
        raise NotImplemented

# A reference to a member of a struct, union, or class that occurs in
# some non-expression context, e.g., a designated initializer.
class Node_MEMBER_REF(Node):
    def __init__(self):
        raise NotImplemented

# A reference to a labeled statement.
class Node_LABEL_REF(Node):
    def __init__(self):
        raise NotImplemented

# A reference to a set of overloaded functions or function templates
# that has not yet been resolved to a specific function or function template.
class Node_OVERLOADED_DECL_REF(Node):
    def __init__(self):
        raise NotImplemented

# A reference to a variable that occurs in some non-expression
# context, e.g., a C++ lambda capture list.
class Node_VARIABLE_REF(Node):
    def __init__(self):
        raise NotImplemented

###
# Invalid/Error Kinds

class Node_INVALID_FILE(Node):
    def __init__(self):
        raise NotImplemented
class Node_NO_DECL_FOUND(Node):
    def __init__(self):
        raise NotImplemented
class Node_NOT_IMPLEMENTED(Node):
    def __init__(self):
        raise NotImplemented
class Node_INVALID_CODE(Node):
    def __init__(self):
        raise NotImplemented

###
# Expression Kinds

# An expression whose specific kind is not exposed via this interface.
#
# Unexposed expressions have the same operations as any other kind of
# expression; one can extract their location information, spelling, children,
# etc. However, the specific kind of the expression is not reported.
class Node_UNEXPOSED_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# An expression that refers to some value declaration, such as a function,
# variable, or enumerator.
class Node_DECL_REF_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# An expression that refers to a member of a struct, union, class, Objective-C
# class, etc.
class Node_MEMBER_REF_EXPR(Node):
    def __init__(self):
        raise NotImplemented


# An expression that sends a message to an Objective-C object or class.
class Node_OBJC_MESSAGE_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# An expression that represents a block literal.
class Node_BLOCK_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# An integer literal.
class Node_INTEGER_LITERAL(Node):
    def __init__(self):
        raise NotImplemented

# A floating point number literal.
class Node_FLOATING_LITERAL(Node):
    def __init__(self):
        raise NotImplemented

# An imaginary number literal.
class Node_IMAGINARY_LITERAL(Node):
    def __init__(self):
        raise NotImplemented

# A string literal.
class Node_STRING_LITERAL(Node):
    def __init__(self):
        raise NotImplemented

# A character literal.
class Node_CHARACTER_LITERAL(Node):
    def __init__(self):
        raise NotImplemented

# A parenthesized expression, e.g. "(1)".
#
# This AST Node is only formed if full location information is requested.
class Node_PAREN_EXPR(Node):
    def __init__(self):
        raise NotImplemented


# Compound assignment such as "+=".
class Node_COMPOUND_ASSIGNMENT_OPERATOR(Node):
    def __init__(self):
        raise NotImplemented


# Describes an C or C++ initializer list.
class Node_INIT_LIST_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# The GNU address of label extension, representing &&label.
class Node_ADDR_LABEL_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# This is the GNU Statement Expression extension(Node):
# ({int X=4; X;})
class Node_StmtExpr(Node):
    def __init__(self):
        raise NotImplemented

# Represents a C11 generic selection.
class Node_GENERIC_SELECTION_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Implements the GNU __null extension, which is a name for a null
# pointer constant that has integral type (e.g., int or long) and is the same
# size and alignment as a pointer.
#
# The __null extension is typically only used by system headers, which classine
# NULL as __null in C++ rather than using 0 (which is an integer that may not
# match the size of a pointer).
class Node_GNU_NULL_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# C++'s static_cast<> expression.
class Node_CXX_STATIC_CAST_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# C++'s dynamic_cast<> expression.
class Node_CXX_DYNAMIC_CAST_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# C++'s reinterpret_cast<> expression.
class Node_CXX_REINTERPRET_CAST_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# C++'s const_cast<> expression.
class Node_CXX_CONST_CAST_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Represents an explicit C++ type conversion that uses "functional"
# notion (C++ [expr.type.conv]).
#
# Example(Node):
    def __init__(self):
        raise NotImplemented
# \code
#   x = int(0.5);
# \endcode
class Node_CXX_FUNCTIONAL_CAST_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# A C++ typeid expression (C++ [expr.typeid]).
class Node_CXX_TYPEID_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# [C++ 2.13.5] C++ Boolean Literal.
class Node_CXX_BOOL_LITERAL_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# [C++0x 2.14.7] C++ Pointer Literal.
class Node_CXX_NULL_PTR_LITERAL_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Represents the "this" expression in C++
class Node_CXX_THIS_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# [C++ 15] C++ Throw Expression.
#
# This handles 'throw' and 'throw' assignment-expression. When
# assignment-expression isn't present, Op will be null.
class Node_CXX_THROW_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# A new expression for memory allocation and constructor calls, e.g(Node):
    def __init__(self):
        raise NotImplemented
# "new CXXNewExpr(foo)".
class Node_CXX_NEW_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# A delete expression for memory deallocation and destructor calls,
# e.g. "delete[] pArray".
class Node_CXX_DELETE_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Represents a unary expression.
class Node_CXX_UNARY_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# ObjCStringLiteral, used for Objective-C string literals i.e. "foo".
class Node_OBJC_STRING_LITERAL(Node):
    def __init__(self):
        raise NotImplemented

# ObjCEncodeExpr, used for in Objective-C.
class Node_OBJC_ENCODE_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# ObjCSelectorExpr used for in Objective-C.
class Node_OBJC_SELECTOR_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Objective-C's protocol expression.
class Node_OBJC_PROTOCOL_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# An Objective-C "bridged" cast expression, which casts between
# Objective-C pointers and C pointers, transferring ownership in the process.
#
# \code
#   NSString *str = (__bridge_transfer NSString *)CFCreateString();
# \endcode
class Node_OBJC_BRIDGE_CAST_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Represents a C++0x pack expansion that produces a sequence of
# expressions.
#
# A pack expansion expression contains a pattern (which itself is an
# expression) followed by an ellipsis. For example(Node):
    def __init__(self):
        raise NotImplemented
class Node_PACK_EXPANSION_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Represents an expression that computes the length of a parameter
# pack.
class Node_SIZE_OF_PACK_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Represents a C++ lambda expression that produces a local function
# object.
#
class Node_LAMBDA_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Objective-c Boolean Literal.
class Node_OBJ_BOOL_LITERAL_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Represents the "self" expression in a ObjC method.
class Node_OBJ_SELF_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP 4.0 [2.4, Array Section].
class Node_OMP_ARRAY_SECTION_EXPR(Node):
    def __init__(self):
        raise NotImplemented

# Represents an @available(...) check.
class Node_OBJC_AVAILABILITY_CHECK_EXPR(Node):
    def __init__(self):
        raise NotImplemented


# A statement whose specific kind is not exposed via this interface.
#
# Unexposed statements have the same operations as any other kind of statement;
# one can extract their location information, spelling, children, etc. However,
# the specific kind of the statement is not reported.
class Node_UNEXPOSED_STMT(Node):
    def __init__(self):
        raise NotImplemented




# A classault statement.
class Node_classAULT_STMT(Node):
    def __init__(self):
        raise NotImplemented


# A do statement.
class Node_DO_STMT(Node):
    def __init__(self):
        raise NotImplemented



# An indirect goto statement.
class Node_INDIRECT_GOTO_STMT(Node):
    def __init__(self):
        raise NotImplemented

# A GNU-style inline assembler statement.
class Node_ASM_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Objective-C's overall @try-@catch-@finally statement.
class Node_OBJC_AT_TRY_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Objective-C's @catch statement.
class Node_OBJC_AT_CATCH_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Objective-C's @finally statement.
class Node_OBJC_AT_FINALLY_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Objective-C's @throw statement.
class Node_OBJC_AT_THROW_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Objective-C's @synchronized statement.
class Node_OBJC_AT_SYNCHRONIZED_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Objective-C's autorealease pool statement.
class Node_OBJC_AUTORELEASE_POOL_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Objective-C's for collection statement.
class Node_OBJC_FOR_COLLECTION_STMT(Node):
    def __init__(self):
        raise NotImplemented

# C++'s catch statement.
class Node_CXX_CATCH_STMT(Node):
    def __init__(self):
        raise NotImplemented

# C++'s try statement.
class Node_CXX_TRY_STMT(Node):
    def __init__(self):
        raise NotImplemented

# C++'s for (* (Node):
class Node_CXX_FOR_RANGE_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Windows Structured Exception Handling's try statement.
class Node_SEH_TRY_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Windows Structured Exception Handling's except statement.
class Node_SEH_EXCEPT_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Windows Structured Exception Handling's finally statement.
class Node_SEH_FINALLY_STMT(Node):
    def __init__(self):
        raise NotImplemented

# A MS inline assembly statement extension.
class Node_MS_ASM_STMT(Node):
    def __init__(self):
        raise NotImplemented

# The null statement.
class Node_NULL_STMT(Node):
    def __init__(self):
        raise NotImplemented

# Adaptor class for mixing declarations with statements and expressions.
class Node_DECL_STMT(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP parallel directive.
class Node_OMP_PARALLEL_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP SIMD directive.
class Node_OMP_SIMD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP for directive.
class Node_OMP_FOR_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP sections directive.
class Node_OMP_SECTIONS_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP section directive.
class Node_OMP_SECTION_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP single directive.
class Node_OMP_SINGLE_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP parallel for directive.
class Node_OMP_PARALLEL_FOR_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP parallel sections directive.
class Node_OMP_PARALLEL_SECTIONS_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP task directive.
class Node_OMP_TASK_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP master directive.
class Node_OMP_MASTER_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP critical directive.
class Node_OMP_CRITICAL_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP taskyield directive.
class Node_OMP_TASKYIELD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP barrier directive.
class Node_OMP_BARRIER_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP taskwait directive.
class Node_OMP_TASKWAIT_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP flush directive.
class Node_OMP_FLUSH_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# Windows Structured Exception Handling's leave statement.
class Node_SEH_LEAVE_STMT(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP ordered directive.
class Node_OMP_ORDERED_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP atomic directive.
class Node_OMP_ATOMIC_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP for SIMD directive.
class Node_OMP_FOR_SIMD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP parallel for SIMD directive.
class Node_OMP_PARALLELFORSIMD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target directive.
class Node_OMP_TARGET_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP teams directive.
class Node_OMP_TEAMS_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP taskgroup directive.
class Node_OMP_TASKGROUP_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP cancellation point directive.
class Node_OMP_CANCELLATION_POINT_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP cancel directive.
class Node_OMP_CANCEL_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target data directive.
class Node_OMP_TARGET_DATA_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP taskloop directive.
class Node_OMP_TASK_LOOP_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP taskloop simd directive.
class Node_OMP_TASK_LOOP_SIMD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP distribute directive.
class Node_OMP_DISTRIBUTE_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target enter data directive.
class Node_OMP_TARGET_ENTER_DATA_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target exit data directive.
class Node_OMP_TARGET_EXIT_DATA_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target parallel directive.
class Node_OMP_TARGET_PARALLEL_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target parallel for directive.
class Node_OMP_TARGET_PARALLELFOR_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target update directive.
class Node_OMP_TARGET_UPDATE_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP distribute parallel for directive.
class Node_OMP_DISTRIBUTE_PARALLELFOR_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP distribute parallel for simd directive.
class Node_OMP_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP distribute simd directive.
class Node_OMP_DISTRIBUTE_SIMD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target parallel for simd directive.
class Node_OMP_TARGET_PARALLEL_FOR_SIMD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP target simd directive.
class Node_OMP_TARGET_SIMD_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

# OpenMP teams distribute directive.
class Node_OMP_TEAMS_DISTRIBUTE_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

###
# Other Kinds


###
# Attributes


class Node_IB_ACTION_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_IB_OUTLET_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_IB_OUTLET_COLLECTION_ATTR(Node):
    def __init__(self):
        raise NotImplemented

class Node_CXX_FINAL_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_CXX_OVERRIDE_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_ANNOTATE_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_ASM_LABEL_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_PACKED_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_PURE_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_CONST_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_NODUPLICATE_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_CUDACONSTANT_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_CUDADEVICE_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_CUDAGLOBAL_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_CUDAHOST_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_CUDASHARED_ATTR(Node):
    def __init__(self):
        raise NotImplemented

class Node_VISIBILITY_ATTR(Node):
    def __init__(self):
        raise NotImplemented

class Node_DLLEXPORT_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_DLLIMPORT_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_CONVERGENT_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_WARN_UNUSED_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_WARN_UNUSED_RESULT_ATTR(Node):
    def __init__(self):
        raise NotImplemented
class Node_ALIGNED_ATTR(Node):
    def __init__(self):
        raise NotImplemented

###
# Preprocessing
class Node_PREPROCESSING_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented
class Node_MACRO_classINITION(Node):
    def __init__(self):
        raise NotImplemented
class Node_MACRO_INSTANTIATION(Node):
    def __init__(self):
        raise NotImplemented
class Node_INCLUSION_DIRECTIVE(Node):
    def __init__(self):
        raise NotImplemented

###
# Extra declaration

# A module import declaration.
class Node_MODULE_IMPORT_DECL(Node):
    def __init__(self):
        raise NotImplemented
# A type alias template declaration
class Node_TYPE_ALIAS_TEMPLATE_DECL(Node):
    def __init__(self):
        raise NotImplemented

# A friend declaration
class Node_FRIEND_DECL(Node):
    def __init__(self):
        raise NotImplemented

# A code completion overload candidate.
class Node_OVERLOAD_CANDIDATE(Node):
    def __init__(self):
        raise NotImplemented


import sys
from clang.cindex import Index
from clang.cindex import CursorKind as CK
from optparse import OptionParser, OptionGroup

parser = OptionParser("usage: %prog [options] {filename} [clang-args*]")
parser.disable_interspersed_args()
(opts, args) = parser.parse_args()
if len(args) == 0:
    parser.error('invalid number arguments')

# FIXME: Add an output file option
out = sys.stdout

index = Index.create(excludeDecls=False)
tu = index.parse(None, args)
if not tu:
    parser.error("unable to load input")

from pprint import pprint

def debug(obj):
    res = {}
    for attr in dir(obj):
        try:
            v = getattr(obj, attr)
            res[attr] = v
        except:
            res[attr] = "IMPOSSIBLE TO COMPUTE"
            pass
        
    pprint(res)
  

ident = 0
def ast_2_ast(node):
    # print(dir(node))
    # print(node.displayname)
    # for child in node.get_children():
    #    traverse(child)
    global ident
    print('   ' * ident + f'{node.kind} {node.spelling}')
    if node.kind == CK.TRANSLATION_UNIT:
        ext = list(map(ast_2_ast, node.get_children()))
        return Node_TRANSLATION_UNIT(ext)
    elif node.kind == CK.TYPEDEF_DECL:
        print('   ' * ident +  f'typedef {node.underlying_typedef_type.kind}, {node.type.spelling};')
        return Typedef(node.type.spelling, "quals", "storage", node.underlying_typedef_type.kind) # TODO
    elif node.kind == CK.STRUCT_DECL:
        name = node.spelling
        print('   ' * ident + f'struct {{')
        ident += 1
        decls = list(map(ast_2_ast, node.get_children()))
        ident -= 1
        print('   ' * ident + f'}}{name};')
        return Node_STRUCT_DECL(name, decls)
    elif node.kind == CK.FIELD_DECL:
        name = node.spelling
        type = node.type.spelling
        decls = [] # TODO list(map(ast_2_ast, node.get_children()))
        return Node_FIELD_DECL(name, type, decls)
    elif node.kind == CK.INTEGER_LITERAL:
        print('   ' * ident + f'{node.kind} {node.spelling} TODO ?')
        return
    elif node.kind == CK.UNION_DECL:
        name = node.spelling
        print( '   ' + f'union {{')
        ident += 1
        decls = list(map(ast_2_ast, node.get_children()))
        ident -= 1
        print('   ' * ident + f'}}{name};')
        return Node_UNION_DECL(name, decls)
    elif node.kind == CK.VAR_DECL:
        name = node.spelling
        type = ast_2_ast(list(node.get_children())[0])
        return Node_VAR_DECL(name, type)
    elif node.kind == CK.TYPE_REF:
        name = node.spelling
        return Node_TYPE_REF(name)
    elif node.kind == CK.FUNCTION_DECL:
        decl = node.spelling
        print('   ' * ident + " ".join(map(lambda x:x.spelling, node.get_tokens())))
        ident += 1
        asts = list(map(ast_2_ast, node.get_children()))
        ident -= 1
        return Node_FUNCTION_DECL(decl, None, asts)
    elif node.kind == CK.UNEXPOSED_ATTR:
        print('   ' * ident + f'{node.kind} {node.spelling}')
        return Node_UNEXPOSED_ATTR()
    elif node.kind == CK.PARM_DECL:
        ident += 1
        asts = list(map(ast_2_ast, node.get_children()))
        ident -= 1        
        return None # TODO
    elif node.kind == CK.ASM_LABEL_ATTR:
        ident += 1
        asts = list(map(ast_2_ast, node.get_children()))
        ident -= 1        
        return None # TODO
    elif node.kind == CK.COMPOUND_STMT and 1:
        ident += 1
        asts = list(map(ast_2_ast, list(node.get_children())))
        ident -= 1
        return None # Node_COMPOUND_STMT(asts)
    elif node.kind == CK.DECL_STMT:
        ident += 1
        asts = list(map(ast_2_ast, list(node.get_children())))
        ident -= 1
        return None
    elif node.kind == CK.FOR_STMT:
        ident += 1
        init, cond, nextt, stmt  = tuple(map(ast_2_ast, list(node.get_children())))
        ident -= 1
        return Node_FOR_STMT(init, cond, nextt, stmt)
    elif node.kind == CK.BINARY_OPERATOR and 0:
        op = None # TODO ?????
        ident += 1
        l, r = tuple(map(ast_2_ast, list(node.get_children())))
        ident -= 1
        return Node_BINARY_OPERATOR(op, l, r)
    else:
        print(dir(node))
        # debug(node)
        print("============= Node & childrens")
        print(node.kind)
        print(list(map(lambda c: c.kind, node.get_children())))
        print("=============")
        print(f'{node.spelling=}')
        print(f'{node.type.spelling=}')
    
        

        print('raw='+"".join(map(lambda x:x.spelling, node.get_tokens())))
        
        raise Exception("Invalid node:", node)

ast_2_ast(tu.cursor)