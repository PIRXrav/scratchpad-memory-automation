from parse import parse
from collections import defaultdict
from subprocess import check_output

from asttools import c_highlight
from asttools import c_to_ast, ast_to_c

from dmamapping import do_memory_mapping


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
    
    # Verify sections
    config = {}
    # Verify DEF
    for line in sections['DEF'].split('\n'):
        s = parse("#define {} {}", line)
        if (s):
            config[s[0]] = s[1]
    # Verify FUN
    cfun = sections['FUN'].split('\n')[0]
    s = parse("void {}(void){}", sections['FUN'])
    if s is None:
        raise Exception("Malformed file: invalid section FUN")
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
    for cfg, val in config.items():
        if not cfg in user_config.keys():
            raise Exception(f"Unspecified configuration: {cfg}")
        config[cfg] = user_config[cfg]
    
    print("=== GENERATE KERNEL ===")
    print(f"{'name':>10}: {kernel_name}")
    for cfg, val in config.items():
        print(f"{cfg:>10}: {val}")

    sectionspp = {k: cppcode(sections[k], config) for k in KERNEL_CODE_SECTIONS}
    sectionsast = {k: c_to_ast(code) for k, code in sectionspp.items()}
    
    for ksec in KERNEL_CODE_SECTIONS:
        print(ksec + ": ")
        print(c_highlight(ast_to_c(sectionsast[ksec])))

    do_memory_mapping(sectionsast['FUN'])
    print(c_highlight(ast_to_c(sectionsast['FUN'])))

kernel_generate('gemv', {'N': 64, 'M': 64})