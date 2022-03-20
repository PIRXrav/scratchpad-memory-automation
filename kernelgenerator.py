__VERSION__ = 'alpha'

from parse import parse
from collections import defaultdict
from subprocess import check_output

from asttools import c_highlight
from asttools import c_to_ast, ast_to_c, stmt_c_to_ast, expr_c_to_ast
from asttools import delc_to_ptr_decl

from asttools import fun_get_name, fun_set_name

from dmamapping import do_memory_mapping
from pycparser import c_ast

import os

import datetime
import sys

def comment_header(title, **kwargs):
    ident = lambda s: s.replace('\n', '\n' + (' ' * 10 + ' | '))
    comment = lambda s: ' * ' + s.replace('\n', '\n * ')
    raw = "\n".join((f"{k:>10} : {ident(str(v))}" for k, v in kwargs.items()))
    return f"/**{title}\n *\n" + comment(raw) + "\n *\n */\n\n"


def cppcode(code, config):
    """Preprocess source code with config defined
    """
    tmpf = '/tmp/sma_cppcode'
    with open(tmpf, 'w') as f:
        f.write(code)
    cpp_defines = " ".join(map(lambda c: f"-D{c[0]}={c[1]}", config.items()))
    return check_output(f'cpp -P {cpp_defines} {tmpf}',
                        shell=True, universal_newlines=True)


def clean_string(s):
    return s.rstrip(' \n\r')
    
def path_of_kernel(kernel_name):
    return f'kernels/{kernel_name}.c'


KERNEL_SEC = ('DEF', 'ARG', 'FUN')
KERNEL_CODE_SEC = ('ARG', 'FUN')


class Kernel:
    def __init__(self, kernel_name, user_config):

        self.kernel_name = kernel_name

        # Get sections
        self.sections = defaultdict(str)
        itersec = iter(KERNEL_SEC)
        current_section = "UNTRACKED"
        with open(path_of_kernel(kernel_name)) as f:
            for line in f:
                s = parse("//***SMA {}", line)
                if(s):
                    current_section = s[0]
                    expected_section = next(itersec)
                    if current_section != expected_section:
                        raise Exception(f"Malformed file: section is {current_section} but should be {expected_section}")
                else:
                    if clean_string(line) != "":
                        self.sections[current_section] += line
        
        # Get config
        self.config = {}
        for line in self.sections['DEF'].split('\n'):
            s = parse("#define {} {}", line)
            if (s):
                self.config[s[0]] = s[1]

        # Set configuration according to kernel configuration
        for cfg, val in self.config.items():
            if not cfg in user_config.keys():
                raise Exception(f"Unspecified configuration: {cfg}")
            self.config[cfg] = user_config[cfg]

        # Key area
        self.sectionspp = {k: cppcode(self.sections[k], self.config) for k in KERNEL_CODE_SEC}
        self.sectionsast = {k: c_to_ast(code) for k, code in self.sectionspp.items()}
        self.fun = self.sectionsast['FUN'].ext[0]
        self.fun_arg_list = []
        self.decls = self.sectionsast['ARG'].ext

        # Verify file        
        if self.fun.__class__ != c_ast.FuncDef:
            raise Exception(f"Malformed file: {self.fun} is not of type FuncDef")

        if self.fun.decl.type.args != None:
            raise Exception(f'Malformed file: function already has arguments')
        self.fun.decl.type.args = c_ast.ParamList(self.fun_arg_list)

        for decl in self.decls:
            if decl.__class__ != c_ast.Decl:
                raise Exception(f"Malformed file: {decl} is not of type Decl")

        for ksec in KERNEL_CODE_SEC:
            print(ksec + ": ")
            print(c_highlight(ast_to_c(self.sectionsast[ksec])))
      

    def get_config(self):
        """Return current kernel configurarion
        """
        return self.config


    def process(self, do_mem_mapping=True):    
        # Memory mapping
        if do_mem_mapping:
            do_memory_mapping(self.fun)

        # Add args
        for decl in self.decls:
            arg_name = decl.name + '_arg'
            ptr_decl = delc_to_ptr_decl(decl)
            ptr_decl.init = c_ast.ID(arg_name)
            # Append arguments
            self.fun.body.block_items.insert(0, ptr_decl)
            self.fun_arg_list.append(expr_c_to_ast(f'void *{arg_name}'))
        
        # Update function name
        fun_name = fun_get_name(self.fun) + ''.join(map(lambda c: '_' + ''.join(map(str, c)), self.config.items()))
        fun_set_name(self.fun, fun_name)

        return self


    def generate_header(self):
        fun_name = fun_get_name(self.fun)
        filename = fun_name + '.h'
        code = ''
        code += comment_header(f' GENERATED FILE: {filename}',
                            Generator=os.path.basename(__file__),
                            Function=fun_name,
                            Version=__VERSION__,
                            Python=sys.version,
                            Date=datetime.datetime.now(),
                            **self.config,
                            **self.sectionspp)
        code += '#include "dma.h"\n'
        code += '\n'
        code += ast_to_c(self.fun)
        return filename, code


    def generate_benchmark(self):
        decl_names = [d.name for d in self.decls]
        fun_name = fun_get_name(self.fun)
        argpp = self.sectionspp['ARG']
        filename = fun_name + '.c'
        code = ''
        code += comment_header(f' GENERATED FILE: {filename}',
                            Generator=os.path.basename(__file__),
                            Benchark=fun_name,
                            Version=__VERSION__,
                            Python=sys.version,
                            Date=datetime.datetime.now(),
                            **self.config)
        code += f'#include "{fun_name}.h"\n'
        code += f'#include <stdio.h>\n'
        code += f'#include <stdint.h>\n'
        code += argpp
        code += f'int main(void){{\n'
        code += f'    printf("Run benchmark {fun_name}\\n");\n'
        code += f'    /* Initialisation */\n'
        code += f'    //printf("Init ...\\n");\n'
        for dn in decl_names:
            code += f'    for (size_t i = 0; i < sizeof({dn}); i++) {{*((char*){dn}+i) = (char)i;}}\n'
        code += f'    /* Run */\n'
        code += f'    //printf("Run ...\\n");\n'
        code += f'    {fun_name}({", ".join(decl_names)});\n'
        code += f'    /* Hash state */\n'
        code += f'    int64_t hash = 0;\n'
        code += f'    //printf("Hash ...\\n");\n'
        for dn in decl_names:
            code += f'    for (size_t i = 0; i < sizeof({dn}); i++) {{hash += *((unsigned char*){dn}+i);}}\n'
        code += f'    printf("%ld\\n", hash);\n'
        code += f'    return 0;'
        code += f'}}\n'
        return filename, code

    def __str__(self):
        return f'{self.kernel_name} {self.config}'


if __name__ == '__main__':
    kernel = Kernel('gemv', {'N': 42, 'M': 64})
    print(kernel)
    kernel.process()

    filename, code = kernel.generate_header()
    print(c_highlight(code))
    filename, code = kernel.generate_benchmark()
    print(c_highlight(code))
    
    with open(filename, 'w') as f:
        f.write(code)
