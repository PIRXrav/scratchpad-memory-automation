from pprint import pprint 
from colorama import Fore, Back, Style

NL = '\n'
QUALTYPE = 'qualType'


class Node:

    inner = []
    def show(self, prefix):
        classs_raw = self.__class__.__name__
        classs_color = Fore.RED
        if classs_raw.endswith('Attr'):
            classs_color = Fore.LIGHTBLUE_EX
        elif classs_raw.endswith('Decl'):
            classs_color = Fore.LIGHTGREEN_EX
        elif classs_raw.endswith('Stmt'):
            classs_color = Fore.LIGHTYELLOW_EX
        classs = Style.BRIGHT + Style.DIM + classs_color + classs_raw + Fore.RESET + Style.RESET_ALL
        name = (Fore.LIGHTBLUE_EX + str(self.name) + Fore.RESET) if hasattr(self, 'name') and self.name != None else ''
        type = (Fore.LIGHTRED_EX + str(self.type.rawdata['qualType']) + Fore.RESET + ' ') if hasattr(self, 'type') and self.type != None else ''
        value = (Fore.LIGHTYELLOW_EX + str(self.value) + Fore.RESET) if hasattr(self, 'value') and self.value != None else ''
        valueCategory = (' [' + Fore.LIGHTWHITE_EX + str(self.valueCategory) + Fore.RESET + '] ') if hasattr(self, 'valueCategory') and self.valueCategory != None else ''
        storageClass = (Fore.LIGHTWHITE_EX + str(self.storageClass) + Fore.RESET + ' ') if hasattr(self, 'storageClass') and self.storageClass != None else ''
        return prefix + f"{classs} > {storageClass}{type}{name}{value}{valueCategory} {self.doc()}\n" + ''.join(map(lambda x: x.show(prefix + ' | '), self.inner))

    def doc(self):
        return ''

    def c(self):
        print(self)
        raise Exception(f'Unimplemented node export: {self.__class__}')

    def cin(self):
        return (n.c() for n in self.inner)

    def __str__(self):
        return self.show('')
    
class NodeVisitor:
    """ A base NodeVisitor class for visiting Nodes.
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
        for c in node.inner:
            self.visit(c)


class TranslationUnitDecl(Node):
    def __init__(self, loc=None, range=None, inner=[]):
        self.loc = loc
        self.range = range
        self.inner = inner
    
    def c(self):
        return ''.join(self.cin())

class TypedefDecl(Node):
    def __init__(self, loc=None, range=None, previousDecl=None, isImplicit=None, isReferenced=None, name=None, type=None, inner=[]):
        self.loc = loc
        self.range = range
        self.previousDecl = previousDecl
        self.isImplicit = isImplicit
        self.isReferenced = isReferenced
        self.name = name
        self.type = type
        self.inner = inner
    
    def c(self):
        return f'typedef {self.type.rawdata[QUALTYPE]} {self.name};'

class BuiltinType(Node):
    def __init__(self, type=None):
        self.type = type

class RecordType(Node):
    def __init__(self, type=None, decl=None):
        self.type = type
        self.decl = decl


class PointerType(Node):
    def __init__(self, type=None, inner=[]):
        self.type = type
        self.inner = inner

class ConstantArrayType(Node):
    def __init__(self, type=None, size=None, inner=[]):
        self.type = type
        self.size = size
        self.inner = inner

class TypedefType(Node):
    def __init__(self, type=None, decl=None, inner=[]):
        self.type = type
        self.decl = decl
        self.inner = inner

class FieldDecl(Node):
    def __init__(self, loc=None, range=None, name=None, type=None):
        self.loc = loc
        self.range = range
        self.name = name
        self.type = type

    def c(self):
        return f'{self.type.rawdata[QUALTYPE]} {self.name};'

class RecordDecl(Node):
    def __init__(self, name=None, loc=None, range=None, tagUsed=None, completeDefinition=None, inner=[], previousDecl=None):
        self.name = name
        self.loc = loc
        self.range = range
        self.tagUsed = tagUsed
        self.completeDefinition = completeDefinition
        self.inner = inner
        self.previousDecl = previousDecl
    
    def c(self):
        return f'struct {{{"".join(self.cin())}}}\n{self.name};\n'

class ElaboratedType(Node):
    def __init__(self, type=None, ownedTagDecl=None, inner=[]):
        self.type = type
        self.ownedTagDecl = ownedTagDecl
        self.inner = inner

class VarDecl(Node):
    def __init__(self, loc=None, range=None, isUsed=None, name=None, mangledName=None, type=None, init=None, storageClass=None, inner=[]):
        self.loc = loc
        self.range = range
        self.isUsed = isUsed
        self.name = name
        self.mangledName = mangledName
        self.type = type
        self.init = init
        self.storageClass = storageClass
        self.inner = inner

    def c(self):
        return f'{self.storageClass if self.storageClass else ""} {self.type.rawdata[QUALTYPE]} {self.name} {" = " + self.inner[0].c() if self.init else ""};'


class FunctionDecl(Node):
    def __init__(self, loc=None, range=None, previousDecl=None, isImplicit=None, isUsed=None, name=None, mangledName=None, type=None, storageClass=None, variadic=None, inner=[]):
        self.loc = loc
        self.range = range
        self.previousDecl = previousDecl
        self.isImplicit = isImplicit
        self.isUsed = isUsed
        self.name = name
        self.mangledName = mangledName
        self.type = type
        self.storageClass = storageClass
        self.variadic = variadic
        self.inner = inner

    def get_dest_type(self):
        return self.type.rawdata[QUALTYPE].split('(', 1)[0]
    
    def c(self):
        nb_args = sum((type(n) is ParmVarDecl for n in self.inner))
        prefix = f'{(self.storageClass + " ") if self.storageClass else ""}{self.get_dest_type()}'
        args = ", ".join((n.c() for n in self.inner[0:nb_args])) if nb_args else 'void'
        if self.variadic:
            args += ', ...'
        body = "".join((n.c() for n in self.inner[nb_args:]))
        return f'{prefix}{self.name}({args}){body};\n' # TODO
    
    def doc(self):
        return f'{self.isImplicit=} {self.isUsed} {self.variadic=} {self.previousDecl=}'
    

class ParmVarDecl(Node):
    def __init__(self, loc=None, range=None, isUsed=None, name=None, mangledName=None, type=None):
        self.loc = loc
        self.range = range
        self.isUsed = isUsed
        self.name = name
        self.mangledName = mangledName
        self.type = type

    def c(self):
        return f'{self.type.rawdata[QUALTYPE]}{(" " + self.name) if self.name else ""}'

class NoThrowAttr(Node):
    def __init__(self, range=None):
        self.range = range
    
    def c(self):
         return '__NO_THROW' # TODO

class RestrictAttr(Node):
    def __init__(self, range=None):
        self.range = range

    def c(self):
         return '__RESTRICT' # TODO

class BuiltinAttr(Node):
    def __init__(self, range=None, inherited=None, implicit=None):
        self.range = range
        self.inherited = inherited
        self.implicit = implicit

    def c(self):
         return '__BUILTIN' # TODO

class FormatAttr(Node):
    def __init__(self, range=None, implicit=None, inherited=None):
        self.range = range
        self.implicit = implicit
        self.inherited = inherited
    
    def c(self):
        return '__FORMAT' # TODO
        
class AsmLabelAttr(Node):
    def __init__(self, range=None):
        self.range = range
    
    def c(self):
        return '__ASM_LABEL' # TODO

class InitListExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, array_filler=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.array_filler = array_filler

    def doc(self):
        return f'{self.array_filler=} {self.valueCategory=}'
        
    def c(self):
        return f'{{{"".join(self.cin())}}}'

class ImplicitValueInitExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory

class ImplicitCastExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, castKind=None, isPartOfExplicitCast=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.castKind = castKind
        self.isPartOfExplicitCast = isPartOfExplicitCast
        self.inner = inner

    def c(self):
        assert len(self.inner) == 1
        return f'{self.inner[0].c()}' # ({self.type.rawdata[QUALTYPE]}) 

class IntegerLiteral(Node):
    def __init__(self, range=None, type=None, valueCategory=None, value=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.value = value
    
    def c(self):
        return str(self.value)

class StringLiteral(Node):
    def __init__(self, range=None, type=None, valueCategory=None, value=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.value = value
    
    def c(self):
        return str(self.value)

class CompoundStmt(Node):
    def __init__(self, range=None, inner=[]):
        self.range = range
        self.inner = inner

    def c(self):
        return f'{{{(NL).join(self.cin())}}}'
    
class DeclStmt(Node):
    def __init__(self, range=None, inner=[]):
        self.range = range
        self.inner = inner

    def c(self):
        assert len(self.inner) == 1
        return f'{self.inner[0].c()}'

class ForStmt(Node):
    def __init__(self, range=None, inner=[]):
        self.range = range
        self.inner = inner

    def c(self):
        print(self)
        # TODO ind 2 ?
        return f'for({self.inner[0].c()} {self.inner[2].c()}; {self.inner[3].c()}) {self.inner[4].c()}'

class BinaryOperator(Node):
    def __init__(self, range=None, type=None, valueCategory=None, opcode=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.opcode = opcode
        self.inner = inner

    def c(self):
        assert len(self.inner) == 2
        return f'{self.inner[0].c()} {self.opcode} {self.inner[1].c()}'

    def doc(self):
        return f'{self.opcode=} {self.valueCategory}'

class UnaryOperator(Node):
    def __init__(self, range=None, type=None, valueCategory=None, isPostfix=None, opcode=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.isPostfix = isPostfix
        self.opcode = opcode
        self.inner = inner

    def c(self):
        assert len(self.inner) == 1
        return f'{self.opcode if not self.isPostfix else ""}{"".join(self.cin())}{self.opcode if self.isPostfix else ""}'

    def doc(self):
        return f'{self.opcode=} {self.valueCategory}'


class DeclRefExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, referencedDecl=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.referencedDecl = referencedDecl # Node

    def c(self):
        return self.referencedDecl.name

    def doc(self):
        return self.referencedDecl.name

class ArraySubscriptExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.inner = inner
    
    def c(self):
        assert len(self.inner) == 2
        return f'{self.inner[0].c()}[{self.inner[1].c()}]'

class CompoundAssignOperator(Node):
    def __init__(self, range=None, type=None, valueCategory=None, opcode=None, computeLHSType=None, computeResultType=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.opcode = opcode
        self.computeLHSType = computeLHSType
        self.computeResultType = computeResultType
        self.inner = inner

    def c(self):
        assert len(self.inner) == 2
        return f'{self.inner[0].c()} = {self.inner[1].c()};'

class CStyleCastExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, castKind=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.castKind = castKind
        self.inner = inner
    
    def c(self):
        assert len(self.inner) == 1
        return f'({self.type.rawdata[QUALTYPE]}){self.inner[0].c()}'

class CallExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.inner = inner
    
    def c(self):
        return f'{self.inner[0].c()}({", ".join((n.c() for n in self.inner[1:]))})'

class ReturnStmt(Node):
    def __init__(self, range=None, inner=[]):
        self.range = range
        self.inner = inner

    def c(self):
        assert len(self.inner) == 1
        return f'return {self.inner[0].c()}'

class AlignedAttr(Node):
    def __init__(self, range=None, inner=None):
        self.range = range
        self.inner = inner

class ConstantExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, value=None, inner=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.value = value
        self.inner = inner

class RawNode(Node):
    def __init__(self, rawdata):
        self.rawdata = rawdata

    def __repr__(self):
        return str(self.rawdata)

    def __str__(self):
        return str(self.rawdata)

    def doc(self):
        return str(self)
    
def json_to_ast(json_node):
    if type(json_node) == list:
        return list(map(json_to_ast, json_node))
    if type(json_node) == bool:
        return json_node
    if type(json_node) == str:
        return json_node
    if type(json_node) == int:
        return json_node
        
    if type(json_node) == dict:
        if 'kind' in json_node.keys(): # Create Node
            node_class = json_node['kind']
            keys = list(json_node.keys())
            keys.remove('id')
            keys.remove('kind')


            try:  
                python_class = eval(node_class)
                try:
                    args = {k: json_to_ast(json_node[k]) for k in keys}
                    return python_class(**args)
                except TypeError:
                    raise NameError("Invalid args call ", node_class)

            except NameError:
                args = ", ".join(map(lambda k: f'{k}=None', keys))
                print(f'class {self_class}(Node):')
                print(f'    def __init__(self, {args}):')
                for key in keys:
                    print(f'        self.{key} = {key}')
                print()
                raise Exception("You have to define the class", node_class)
             
            
        else:
            return RawNode(json_node) # json_node

    raise Exception("Unknown json type", type(json_node))

