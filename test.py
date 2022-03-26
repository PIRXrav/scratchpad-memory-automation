import subprocess
from kernelgenerator import Kernel


from isort import file

PREFIX = '/tmp/'
SMA_SOURCE = PREFIX+'sma_source.c'
SMA_BIN = PREFIX+'sma_bin'

CC = 'gcc'
CFLAGS = '-Wall -Wextra -Werror -Idmasimulator -g'
LDFLAGS = ''

def write_file(filename, code):
    with open(filename, 'w') as f:
        f.write(code)

from asttools import c_highlight
import sys

def shell(cmd):
    print(cmd)
    return subprocess.call(cmd, shell=True)


def test_kernel(do_mem_mapping=True):
    # Generate C code    
    kernel = Kernel('gemv', {'N': 42, 'M': 64})
    kernel.process(do_mem_mapping=do_mem_mapping)

    hname, hcode = kernel.generate_header()
    print(c_highlight(hcode))
    cname, ccode = kernel.generate_benchmark()
    print(c_highlight(ccode))
    gdbname, gdbcode, dumpfiles = kernel.generate_benchmark_gdb(prefix=PREFIX+('dma_' if do_mem_mapping else ''))
    print(gdbcode)

    hname = PREFIX + hname
    cname = PREFIX + cname
    gdbname = PREFIX + gdbname
    binname = PREFIX + 'sma_bin' + ('_mem_mapping' if do_mem_mapping else '')
    write_file(hname, hcode)
    write_file(cname, ccode)
    write_file(gdbname, gdbcode)

    cmd = f'{CC} {CFLAGS} {LDFLAGS} {cname} -I{PREFIX} -o {binname}'
    ret = shell(cmd)
    assert ret == 0
    # Run
    cmd = f'gdb --batch --command={gdbname} --args {binname}'
    ret = shell(cmd)
    assert ret == 0
    shell(f'wc -c {" ".join(dumpfiles)}')
    return dumpfiles
    # dump binary memory file.bin weights (char*)weights + sizeof(weights)

import filecmp

if __name__ == "__main__":
    files = [test_kernel(do_mem_mapping=test) for test in (False, True)]
    total_diff = 0
    for ff in zip(*tuple(files)):
        eq = filecmp.cmp(*ff)
        total_diff += not eq
        print(f'{ff[0]} and {ff[1]} {"are equals" if eq else "differ"}')
    
    if total_diff:
        raise Exception("Result differ")

    exit(total_diff)

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
