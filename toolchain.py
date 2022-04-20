import subprocess
from ctools import c_highlight

def write_file(filename, code):
    """write file filename with code as data"""
    with open(filename, "w") as f:
        f.write(code)

def read_file(filename):
    with open(filename) as f:
        return f.read()

def write_c_file(filename, code, display_code=False):
    """write file filename with code as data
    """
    write_file(filename, code)
    res = shell(f"clang-format -i {filename}")
    if display_code:  # Display code after clang-format
        print(c_highlight(read_file(filename)))


def shell(cmd, verbose=False):
    """Shell cmd wrapper"""
    if verbose:
        print(cmd)
    try:
        res = subprocess.check_output(cmd, shell=True).decode('UTF-8')
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.output.decode())
        raise Exception("error")

    if verbose:
        print(res)
    return res


def cpp(code, config):
    """Python CPP wrapper"""
    tmpf = "/tmp/sma_cpp"
    write_file(tmpf, code)
    cpp_defines = " ".join(map(lambda c: f"-D{c[0]}={c[1]}", config.items()))
    return subprocess.check_output(
        f"cpp -P {cpp_defines} {tmpf}", shell=True, universal_newlines=True
    )


OPTIMISE = "O1"
CC = "gcc"
CFLAGS = f"-Wall -Wextra -Werror -Idmasimulator -I. -g -{OPTIMISE}"
LDFLAGS = ""

def gcc(cfiles, binfile, opts="", verbose=False):
    cmd = f"{CC} {CFLAGS} {opts} {LDFLAGS} {' '.join(cfiles)} -I. -o {binfile}"
    shell(cmd, verbose=verbose)

def benchmarkrun(cmd, verbose=False):
    """ All benchmarks must print line:
    PYTHON_RES = {"hash": -450, "hasht": -450, "err": 0}
    """
    out = shell(cmd, verbose=verbose)
    # Process result
    for line in out.split('\n'):
        if 'PYTHON_RES' in line:
            PYTHON_RES = eval(line.split('=')[1])
    return PYTHON_RES, out
