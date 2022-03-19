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

def path_of_kernel(kernel_name):
    return f'kernels/{kernel_name}.c'


KERNEL_SECTIONS = ('DEF', 'ARG', 'FUN')
KERNEL_CODE_SECTIONS = ('ARG', 'FUN')


def kernel_get_sections(kernel_name):
    def clean_string(s):
        return s.rstrip(' \n\r')
    
    sections = defaultdict(str)
    itersec = iter(KERNEL_SECTIONS)
    current_section = "UNTRACKED"

    # Get sections
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
                    sections[current_section] += line
    
    # Get config
    config = {}
    for line in sections['DEF'].split('\n'):
        s = parse("#define {} {}", line)
        if (s):
            config[s[0]] = s[1]

    return sections, config


def cppcode(code, config):
    """Preprocess source code with config defined
    """
    tmpf = '/tmp/sma_cppcode'
    with open(tmpf, 'w') as f:
        f.write(code)
    cpp_defines = " ".join(map(lambda c: f"-D{c[0]}={c[1]}", config.items()))
    return check_output(f'cpp -P {cpp_defines} {tmpf}',
                        shell=True, universal_newlines=True)


def kernel_generate(kernel_name, user_config):
    sections, config = kernel_get_sections(kernel_name)
    
    sectionspp = {k: cppcode(sections[k], config) for k in KERNEL_CODE_SECTIONS}
    sectionsast = {k: c_to_ast(code) for k, code in sectionspp.items()}
    
    # Verify file
    for cfg, val in config.items():
        if not cfg in user_config.keys():
            raise Exception(f"Unspecified configuration: {cfg}")
        config[cfg] = user_config[cfg]
    
    fun = sectionsast['FUN'].ext[0]
    if fun.__class__ != c_ast.FuncDef:
        raise Exception(f"Malformed file: {fun} is not of type FuncDef")

    if fun.decl.type.args != None:
        raise Exception(f'Malformed file: function already has arguments')
    fun_arg_list = []
    fun.decl.type.args = c_ast.ParamList(fun_arg_list)

    decls = sectionsast['ARG'].ext
    for decl in decls:
        if decl.__class__ != c_ast.Decl:
            raise Exception(f"Malformed file: {decl} is not of type Decl")

    for ksec in KERNEL_CODE_SECTIONS:
        print(ksec + ": ")
        print(c_highlight(ast_to_c(sectionsast[ksec])))

    # Memory mapping
    do_memory_mapping(fun)

    # Add args
    for decl in decls:
        arg_name = decl.name + '_arg'
        ptr_decl = delc_to_ptr_decl(decl)
        ptr_decl.init = c_ast.ID(arg_name)
        # Append arguments
        fun.body.block_items.insert(0, ptr_decl)
        fun_arg_list.append(expr_c_to_ast(f'void *{arg_name}'))
    
    # Update function name
    fun_name = fun_get_name(fun) + ''.join(map(lambda c: '_' + ''.join(map(str, c)), config.items()))
    fun_set_name(fun, fun_name)

    return sectionsast['FUN'], sectionspp, config, fun_name


def comment_header(title, **kwargs):
    ident = lambda s: s.replace('\n', '\n' + (' ' * 10 + ' | '))
    comment = lambda s: ' * ' + s.replace('\n', '\n * ')
    raw = "\n".join((f"{k:>10} : {ident(str(v))}" for k, v in kwargs.items()))
    return f"/**{title}\n *\n" + comment(raw) + "\n *\n */\n\n"


import datetime
import sys

def kernel_generate_header(kernel_name, user_config):
    fun, sectionspp, config, fun_name = kernel_generate(kernel_name, user_config)
    filename = fun_name + '.h'
    code = ''
    code += comment_header(f' GENERATED FILE: {filename}',
                           Function=fun_name,
                           Version=__VERSION__,
                           Python=sys.version,
                           Date=datetime.datetime.now(),
                           **config,
                           **sectionspp)
    code += '#include "dma.h"\n'
    code += '\n'
    code += ast_to_c(fun)
    return filename, code

if __name__ == '__main__':
    filename, code = kernel_generate_header('gemv', {'N': 42, 'M': 64})
    print(c_highlight(code))
    with open(filename, 'w') as f:
        f.write(code)
