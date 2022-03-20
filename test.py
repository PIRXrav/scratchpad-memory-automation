import subprocess
from kernelgenerator import Kernel


from isort import file

PREFIX = '/tmp/'
SMA_SOURCE = PREFIX+'sma_source.c'
SMA_BIN = PREFIX+'sma_bin'

CC = 'gcc'
CFLAGS = '-Wall -Wextra -Werror -Idmasimulator -g'
LDFLAGS = ''

def write_c_file(filename, code):
    with open(filename, 'w') as f:
        f.write(code)

from asttools import c_highlight
import sys

def main(do_mem_mapping=True):
    # Generate C code    
    kernel = Kernel('gemv', {'N': 42, 'M': 64})
    kernel.process(do_mem_mapping=do_mem_mapping)

    hname, hcode = kernel.generate_header()
    print(c_highlight(hcode))
    cname, ccode = kernel.generate_benchmark()
    print(c_highlight(ccode))
    
    hname = PREFIX + hname
    cname = PREFIX + cname

    write_c_file(hname, hcode)
    write_c_file(cname, ccode)
    
    cmd = f'{CC} {CFLAGS} {LDFLAGS} {cname} -I{PREFIX} -o {SMA_BIN}'
    ret = subprocess.call(cmd, shell=True)
    assert ret == 0
    # Run
    cmd = f'{SMA_BIN}'
    ret = subprocess.call(cmd, shell=True)
    assert ret == 0

if __name__ == "__main__":
    main(do_mem_mapping=False)
    main(do_mem_mapping=True)
    
    exit(0)

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
