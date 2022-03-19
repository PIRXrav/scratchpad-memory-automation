import subprocess

from isort import file

PREFIX = '/tmp/'
SMA_SOURCE = PREFIX+'sma_source.c'
SMA_BIN = PREFIX+'sma_bin'

CC = 'gcc'
CFLAGS = '-Wall -Wextra -g -Idmasimulator'
LDFLAGS = ''


def compile_and_run_file(file):
    # Compile
    cmd = f'{CC} {CFLAGS} {LDFLAGS} {file} -o {SMA_BIN}'
    ret = subprocess.call(cmd, shell=True)
    assert ret == 0
    # Run
    cmd = f'{SMA_BIN}'
    ret = subprocess.call(cmd, shell=True)
    return ret

def compile_and_run(ccode):
    # Write file
    with open(SMA_SOURCE, 'w') as out_file:
        out_file.write(ccode)
    # Compile and run
    return compile_and_run_file(SMA_SOURCE)



from asttools import ast_to_c
from asttools import c_highight
import sys

def main(filename):
    # Generate C code
    
    funast = kernel_generate('gemv', {'N': 64, 'M': 64})
    print(c_highlight(ast_to_c(funast)))

    ast = do_memory_mapping()
    code = ast_to_c(ast)
    # Append DMA Header
    code = '#include "dma.h"\n' + code

    print(c_highight(code))

    compile_and_run(code)
    compile_and_run_file(filename)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
