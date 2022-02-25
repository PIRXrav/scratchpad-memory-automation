import sys
import json
from astdumpparser import json_to_ast, NodeVisitor


class ConstantVisitor(NodeVisitor):
    def __init__(self):
        self.values = []

    def visit_IntegerLiteral(self, node):
        self.values.append(node.value)



class ExporterVisitor(NodeVisitor):
    def __init__(self):
        self.values = []

    def V(self, node):
        return (self.visit(n) for n in node.inner)
    
    def VNL(self, node):
        return '\n'.join(self.visit(n) for n in node.inner)

   
    def generic_visit(self, node):
        raise Exception(f'Unimplemented node export: {node.__class__}')

# Opening JSON file
f = open('res.json')
# a dictionary
data = json.load(f)
f.close()


ast = json_to_ast(data)
print(ast)


cv = ConstantVisitor()
cv.visit(ast)
print(cv.values)

print(ast.c())