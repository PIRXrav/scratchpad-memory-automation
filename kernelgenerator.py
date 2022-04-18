__VERSION__ = "alpha"

from parse import parse
from collections import defaultdict
from itertools import chain
from subprocess import check_output

from asttools import c_highlight
from asttools import c_to_ast, ast_to_c, expr_c_to_ast
from asttools import fun_get_name, fun_set_name
import asttools as at
import outpreprocessor as opp

from dmamapping import do_memory_mapping, c_ast_arraydecl_to_l

from pycparser import c_ast

import os

import datetime
import sys

import toolchain as tc
import ctools

from copy import deepcopy

def path_of_kernel(kernel_name):
    return f"kernels/{kernel_name}.c"


KERNEL_SEC = ("DEF", "ARG", "FUN")
KERNEL_CODE_SEC = ("ARG", "FUN")


def kernel_compute_name(name, config):
    return name + "".join(map(lambda c: "_" + "".join(map(str, c)), config.items()))

from functools import wraps
from time import time

timing_stack = []

def timing(func):
    """Measure elapsed time"""

    @wraps(func)
    def wrap(*args, **kw):
        global timing_stack
        ts = time()
        result = func(*args, **kw)
        te = time()
        elapsed = int((te - ts) * 1000)
        timing_stack.append(f"func:{func.__name__:>20} took: {elapsed:>8} ms")
        return result

    return wrap

def timing_dump():
    """Display results of timing func"""
    global timing_stack
    for s in timing_stack:
        print(s)
    timing_stack = []


def c_ast_cdecl_to_ast(line):
    s = parse("{} {}", line)
    ctype = s[0].replace(' ', '')
    cdecl = s[1].replace(' ', '')
    if ctype in ('char', 'int', 'float', 'double'):  # TODO all std c types
        decl = c_to_ast(line).ext[0]
    else:
        ext = c_to_ast(f"typedef int {ctype}; {line}").ext
        assert len(ext) == 2
        assert ext[0].__class__ == c_ast.Typedef
        decl = ext[1]
    if decl.__class__ != c_ast.Decl:
        raise Exception(f"Malformed file: {decl} is not of type Decl")
    return decl


class Kernel:
    def __init__(self, kernel_name, user_config):

        self.kernel_name = kernel_name

        # Get sections
        self.sections = defaultdict(str)
        itersec = iter(KERNEL_SEC)
        current_section = "UNTRACKED"
        with open(path_of_kernel(kernel_name)) as f:
            for line in f:
                s = parse("//***SMA {}", line.rstrip(" \n\r"))
                if s:
                    current_section = s[0]
                    expected_section = next(itersec)
                    if current_section != expected_section:
                        raise Exception(
                            f"Malformed file: section is {current_section} but should be {expected_section}"
                        )
                else:
                    if line.rstrip(" \n\r") != "":
                        self.sections[current_section] += line

        # Get config
        self.config = {}
        for line in self.sections["DEF"].split("\n"):
            s = parse("#define {} {}", line)
            if s:
                self.config[s[0]] = s[1]

        # Set configuration according to kernel configuration
        for cfg, val in self.config.items():
            if cfg not in user_config.keys():
                raise Exception(f"Unspecified configuration: {cfg}")
            self.config[cfg] = user_config[cfg]

        # Key area
        self.sectionspp = {
            k: tc.cpp(self.sections[k], self.config) for k in KERNEL_CODE_SEC
        }

        # Process Function ast
        self.fun = c_to_ast(self.sectionspp['FUN']).ext[0]
        self.fun_arg_list = []
        if self.fun.__class__ != c_ast.FuncDef:
            raise Exception(f"Malformed file: {self.fun} is not of type FuncDef")
        if self.fun.decl.type.args is not None:
            raise Exception("Malformed file: function already has arguments")
        self.fun.decl.type.args = c_ast.ParamList(self.fun_arg_list)

        # Process Arguments
        self.decls = []
        for line in self.sectionspp["ARG"].split("\n"):
            s = parse("{} {}", line)
            if s:
                decl = c_ast_cdecl_to_ast(line)
                self.decls.append(decl)

        # Compute memory size
        self.decl_l_namespace = {}
        for decl in self.decls:
            name, ast_type, asts, decl_l = c_ast_arraydecl_to_l(decl)
            self.decl_l_namespace[name] = (ast_type, decl_l)

        self.decl_l_namespace_reference = deepcopy(self.decl_l_namespace)

    def get_config(self):
        """Return current kernel configurarion"""
        return self.config

    def process(self, do_mem_mapping=True):
        # Memory mapping
        if do_mem_mapping:
            do_memory_mapping(self.fun, self.decl_l_namespace)

        # Add args
        for decl_name, (ast_type, l) in self.decl_l_namespace.items():
            arg_name = decl_name + '_arg'
            ram_name = decl_name + "".join(reversed(list((f"[{i}]" for i in l[1:]))))
            code = f'{at.ast_to_c(ast_type)} {ram_name};\n'
            decl = c_ast_cdecl_to_ast(code)
            ptr_decl = at.c_ast_delc_to_ptr_decl(decl)
            ptr_decl.init = c_ast.ID(arg_name)
            at.c_ast_decl_type_add_prefix(ptr_decl, f'__SMA_RAM_PTR{opp.OPP_SPACE}')
            # Append arguments
            self.fun.body.block_items.insert(0, ptr_decl)
            self.fun_arg_list.append(expr_c_to_ast(f"__SMA_RAM_PTR{opp.OPP_SPACE}void *{arg_name}"))

        # Update function name
        fun_name = kernel_compute_name(self.kernel_name, self.config)
        fun_set_name(self.fun, fun_name)

        return self

    def generate_header(self):
        fun_name = fun_get_name(self.fun)
        filename = fun_name + ".h"
        code = ""
        code += ctools.comment_header(
            f" GENERATED FILE: {filename}",
            Generator=os.path.basename(__file__),
            Function=fun_name,
            Version=__VERSION__,
            Python=sys.version,
            Date=datetime.datetime.now(),
            **self.config,
            **self.sectionspp,
        )
        code += '#include "dma.h"\n'
        code += "\n"
        code += opp.opp(ast_to_c(self.fun))
        return filename, code

    def generate_benchmark_gdb(self, prefix=""):
        decl_names = list(chain((d.name for d in self.decls),
                                (d.name + '_test' for d in self.decls)))

        dump_file_names = [prefix + d + ".bin" for d in decl_names]
        print(dump_file_names)

        fun_name = fun_get_name(self.fun)
        filename = fun_name + ".gdb"
        code = ""
        code += "b postbench\n"
        code += "run\n"
        for dn, dump_file in zip(decl_names, dump_file_names):
            code += f"dump binary memory {dump_file} {dn} (char*){dn} + sizeof({dn})\n"
        code += "c\n"
        return filename, code, dump_file_names

    def gen_hash_code(self, decl_name, decl_l):
        indx = [(chr(i + ord("a")), l) for i, l in enumerate((decl_l[1:]))]
        print(indx)
        adr = "".join(list((f"[{i}]" for i, l in reversed(indx))))
        code = ""
        code += f'hash += {decl_name}{adr};\n'
        code += f'hash_test += {decl_name}_test{adr};\n'
        code += f'nb_err += ({decl_name}{adr} != {decl_name}_test{adr});\n'
        for i, l in indx:
            code = f'for(int {i} = 0; {i} < {l}; {i}++) {{\n{code}\n}}\n'
        return code

    def gen_init_code(self, decl_name, decl_l):
        indx = [(chr(i + ord("a")), l) for i, l in enumerate((decl_l[1:]))]
        print(indx)
        adr = "".join(list((f"[{i}]" for i, l in reversed(indx))))     
        code = ""
        for dn in [decl_name, decl_name + '_test']:
            code += f'{dn}{adr} = {"+".join((i for i, l in indx))};\n'
        for i, l in indx:
            code = f'for(int {i} = 0; {i} < {l}; {i}++) {{\n{code}\n}}\n'
        return code

    def generate_benchmark(self):
        import outpreprocessor as opp
        decl_names = [d.name for d in self.decls]
        decl_names_test = map(lambda s: s + "_test", decl_names)
        fun_name = fun_get_name(self.fun)
        argpp = self.sectionspp["ARG"]
        filename = fun_name + ".c"
        code = ""
        code += ctools.comment_header(
            f" GENERATED FILE: {filename}",
            Generator=os.path.basename(__file__),
            Benchark=fun_name,
            Version=__VERSION__,
            Python=sys.version,
            Date=datetime.datetime.now(),
            **self.config,
        )
        code += f'#include "{fun_name}.h"\n'
        code += "#include <stdio.h>\n"
        code += "#include <stdint.h>\n"
        # RAM data
        for decl_name, (ast_type, l) in self.decl_l_namespace.items():
            ram_name = decl_name + '_test' + "".join(reversed(list((f"[{i}]" for i in l[1:]))))
            code += f"{f'__SMA_RAM '+ast_type.type.names[-1]} {ram_name};\n"
        # RAM data reference
        for decl_name, (ast_type, l) in self.decl_l_namespace_reference.items():
            ram_name = decl_name + "".join(reversed(list((f"[{i}]" for i in l[1:]))))
            code += f'{" ".join(ast_type.type.names)} {ram_name};\n'

        code += "int64_t postbench(void){\n"
        code += "    int64_t hash_test = 0;\n"
        code += "    int64_t hash = 0;\n"
        code += "    int64_t nb_err = 0;\n"
        for decl_name, (ast_type, decl_l) in self.decl_l_namespace_reference.items():
            code += self.gen_hash_code(decl_name, decl_l)
            # code += f"    for (size_t i = 0; i < sizeof({dn}); i++) {{hash += *((unsigned char*){dn}+i);}}\n"
        code += '    printf("PYTHON_RES = {\\"hash\\": %ld, \\"hasht\\": %ld, \\"err\\": %ld}\\n", hash, hash_test, nb_err);\n'
        code += '    return nb_err ? 42 : 0;\n'
        code += "}\n"
        code += "void prebench(void){\n"
        for decl_name, (ast_type, decl_l) in self.decl_l_namespace_reference.items():
            code += self.gen_init_code(decl_name, decl_l)
            code += f'    printf("{decl_name} = %p\\n", {decl_name});'
            decl_name += '_test'
            code += f'    printf("{decl_name} = %p\\n", {decl_name});'

        code += "}\n"
        code += "int main(void){{\n"
        code += f'    printf("Run benchmark {fun_name}\\n");\n'
        code += "    /* prebench */\n"
        code += '    printf("prebench ...\\n");\n'
        code += "    prebench();"
        code += "    /* Run */\n"
        code += '    printf("Run ...\\n");\n'
        code += f'    {fun_name}({", ".join(decl_names_test)});\n'
        code += f"{ast_to_c(c_to_ast(self.sectionspp['FUN']).ext[0].body)}"
        code += "    /* postbench */\n"
        code += '    printf("postbench ...\\n");\n'
        code += "    exit(postbench());\n"
        code += "}}\n"
        print(at.c_highlight(code))
        return filename, code

    def __str__(self):
        return f"{self.kernel_name} {self.config}"

    def bench(self):
        """Benchmark a kernel
        """

        PREFIX = "/tmp/"
        SMA_SOURCE = PREFIX + "sma_source.c"
        SMA_BIN = PREFIX + "sma_bin"

        CC = "gcc"
        CFLAGS = "-Wall -Wextra -Werror -Idmasimulator -g -O1"
        LDFLAGS = ""

        @timing
        def generate():
            # Generate C code
            hname, hcode = self.generate_header()
            print(c_highlight(hcode))
            cname, ccode = self.generate_benchmark()
            # print(c_highlight(ccode))
            gdbname, gdbcode, dumpfiles = self.generate_benchmark_gdb(
                prefix=PREFIX
            )
            return (hname, hcode, cname, ccode, gdbname, gdbcode), dumpfiles

        @timing
        def write_files(hname, hcode, cname, ccode, gdbname, gdbcode):
            hname = PREFIX + hname
            cname = PREFIX + cname
            gdbname = PREFIX + gdbname
            tc.write_file(hname, hcode)
            tc.write_file(cname, ccode)
            tc.write_file(gdbname, gdbcode)
            return hname, cname, gdbname

        @timing
        def build(cnames, binname):
            files = " ".join(cnames)
            cmd = f"{CC} {CFLAGS} {LDFLAGS} {files} -I{PREFIX} -o {binname}"
            ret = tc.shell(cmd)
            return ret

        @timing
        def run_simu(gdbname, binname):
            # Run
            cmd = f"gdb --batch --command={gdbname} --args {binname}"
            # cmd = binname
            res = tc.shell(cmd, verbose=True)
            # Process result
            for line in res.split('\n'):
                if 'PYTHON_RES' in line:
                    PYTHON_RES = eval(line.split('=')[1])
                    print(PYTHON_RES)
            return int(PYTHON_RES['err'])

        self.process(do_mem_mapping=True)
        binname = PREFIX + "sma_bin"
        files_plus_code, dumpfiles = generate()
        hname, cname, gdbname = write_files(*files_plus_code)
        build([cname], binname)
        ret = run_simu(gdbname, binname)
        tc.shell(f'wc -c {" ".join(dumpfiles)}')
        timing_dump()
        return ret, dumpfiles


if __name__ == "__main__":
    kernel = Kernel("conv2d", {"X": 16, "Y": 16, "DKX": 3, "DKY": 3})
    print(kernel)
    kernel.process()

    filename, code = kernel.generate_header()
    print(c_highlight(code))
    filename, code = kernel.generate_benchmark()
    print(c_highlight(code))

    with open(filename, "w") as f:
        f.write(code)
