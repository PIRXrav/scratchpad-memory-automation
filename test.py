import subprocess

PREFIX = '/tmp/'
SMA_SOURCE = PREFIX+'sma_source.c'
SMA_BIN = PREFIX+'sma_bin'

CC = 'gcc'
CFLAGS = '-Wall -Wextra -g -Idmasimulator'
LDFLAGS = ''

def compile_and_run(ccode):
    # Write file
    with open(SMA_SOURCE, 'w') as out_file:
        out_file.write(ccode)
    # Compile
    cmd = f'{CC} {CFLAGS} {LDFLAGS} {SMA_SOURCE} -o {SMA_BIN}'
    ret = subprocess.call(cmd, shell=True)
    assert ret == 0
    # Run
    cmd = f'{SMA_BIN}'
    ret = subprocess.call(cmd, shell=True)
    return ret


from analyser import AstToolkit
from analyser import c_highight
import sys

def main(filename):
    # Generate C code
    ast = AstToolkit(filename)
    ast.do_memory_mapping()
    ccode = ast.exportc()
    # Append DMA Header
    
    ccode = '#include "dma.h"\n' + ccode
    print(c_highight(ccode))
    compile_and_run(ccode)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
