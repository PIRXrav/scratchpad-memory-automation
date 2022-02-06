import sys
import json
from pprint import pprint 


# Opening JSON file
f = open('res.json')
# a dictionary
data = json.load(f)
f.close()

# print(data)

from colorama import Fore, Back, Style


class Node:

    inner = []

    def show(self, prefix):
        classs = Style.BRIGHT + Style.DIM + Fore.GREEN + self.__class__.__name__ + Fore.RESET + Style.RESET_ALL
        name = (Fore.LIGHTBLUE_EX + str(self.name) + Fore.RESET) if hasattr(self, 'name') and self.name != None else ''
        type = (Fore.LIGHTRED_EX + str(self.type.rawdata) + Fore.RESET) if hasattr(self, 'type') and self.type != None else ''
        value = (Fore.LIGHTYELLOW_EX + str(self.value) + Fore.RESET) if hasattr(self, 'value') and self.value != None else ''
        valueCategory = (Fore.LIGHTWHITE_EX + str(self.valueCategory) + Fore.RESET) if hasattr(self, 'valueCategory') and self.valueCategory != None else ''
        return prefix + f"{classs} > {type} {name}{value} {valueCategory} {self.doc()}\n" + ''.join(map(lambda x: x.show(prefix + ' | '), self.inner))

    def doc(self):
        return ''

    def __str__(self):
        return self.show('')
        


class TranslationUnitDecl(Node):
    def __init__(self, loc=None, range=None, inner=[]):
        self.loc = loc
        self.range = range
        self.inner = inner

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

class BuiltinType(Node):
    def __init__(self, type=None):
        self.type = type

class RecordType(Node):
    def __init__(self, type=None, decl=None):
        self.type = type
        self.decl = decl

class RecordDecl(Node):
    def __init__(self, name=None):
        self.name = name

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

class RecordDecl(Node):
    def __init__(self, name=None, loc=None, range=None, tagUsed=None, completeDefinition=None, inner=[], previousDecl=None):
        self.name = name
        self.loc = loc
        self.range = range
        self.tagUsed = tagUsed
        self.completeDefinition = completeDefinition
        self.inner = inner
        self.previousDecl = previousDecl

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
    

class ParmVarDecl(Node):
    def __init__(self, loc=None, range=None, name=None, mangledName=None, type=None):
        self.loc = loc
        self.range = range
        self.name = name
        self.mangledName = mangledName
        self.type = type


class NoThrowAttr(Node):
    def __init__(self, range=None):
        self.range = range

class RestrictAttr(Node):
    def __init__(self, range=None):
        self.range = range

class BuiltinAttr(Node):
    def __init__(self, range=None, inherited=None, implicit=None):
        self.range = range
        self.inherited = inherited
        self.implicit = implicit

class FormatAttr(Node):
    def __init__(self, range=None, implicit=None, inherited=None):
        self.range = range
        self.implicit = implicit
        self.inherited = inherited
        
class AsmLabelAttr(Node):
    def __init__(self, range=None):
        self.range = range

class InitListExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, array_filler=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.array_filler = array_filler

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

class IntegerLiteral(Node):
    def __init__(self, range=None, type=None, valueCategory=None, value=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.value = value

class StringLiteral(Node):
    def __init__(self, range=None, type=None, valueCategory=None, value=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.value = value

class CompoundStmt(Node):
    def __init__(self, range=None, inner=[]):
        self.range = range
        self.inner = inner

class DeclStmt(Node):
    def __init__(self, range=None, inner=[]):
        self.range = range
        self.inner = inner

class ForStmt(Node):
    def __init__(self, range=None, inner=[]):
        self.range = range
        self.inner = inner

class BinaryOperator(Node):
    def __init__(self, range=None, type=None, valueCategory=None, opcode=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.opcode = opcode
        self.inner = inner

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

    def doc(self):
        return f'{self.opcode=} {self.valueCategory}'


class DeclRefExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, referencedDecl=None):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.referencedDecl = referencedDecl

    def doc(self):
        return self.referencedDecl.name

class ArraySubscriptExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.inner = inner

class CompoundAssignOperator(Node):
    def __init__(self, range=None, type=None, valueCategory=None, opcode=None, computeLHSType=None, computeResultType=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.opcode = opcode
        self.computeLHSType = computeLHSType
        self.computeResultType = computeResultType
        self.inner = inner

class CStyleCastExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, castKind=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.castKind = castKind
        self.inner = inner

class CallExpr(Node):
    def __init__(self, range=None, type=None, valueCategory=None, inner=[]):
        self.range = range
        self.type = type
        self.valueCategory = valueCategory
        self.inner = inner

class ReturnStmt(Node):
    def __init__(self, range=None, inner=[]):
        self.range = range
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
                print(f'class {node_class}(Node):')
                print(f'    def __init__(self, {args}):')
                for key in keys:
                    print(f'        self.{key} = {key}')
                print()
                raise Exception("You have to define the class", node_class)
             
            
        else:
            return RawNode(json_node) # json_node

    raise Exception("Unknown json type", type(json_node))

ast = json_to_ast(data)
print(ast)


