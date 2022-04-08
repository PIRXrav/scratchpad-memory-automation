
import logging as log
import logging
from colorlog import ColoredFormatter

DEBUG_MODE = 1

if __name__ == '__main__':
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.ERROR
    DEBUG_MODE = 0

LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s %(filename)-16s:%(lineno)-4d>> %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

log = logging.getLogger()

from isort import file
import subprocess
from kernelgenerator import Kernel, kernel_compute_name
import filecmp
from asttools import c_highlight
import sys
import unittest
from ddt import ddt, data

PREFIX = "/tmp/"
SMA_SOURCE = PREFIX + "sma_source.c"
SMA_BIN = PREFIX + "sma_bin"

CC = "gcc"
CFLAGS = "-Wall -Wextra -Werror -Idmasimulator -g -O1"
LDFLAGS = ""


from functools import wraps
from time import time


timing_stack = []
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        global timing_stack
        ts = time()
        result = f(*args, **kw)
        te = time()
        timing_stack.append(f'func:{f.__name__:>20} took: {int((te-ts)*1000):>8} ms')
        # log.info(f'func:{f.__name__} args:[{args}, {kw}] took: %{te-ts} sec')
        return result
    return wrap if DEBUG_MODE else f


def timing_dump():
    global timing_stack
    for s in timing_stack:
        log.info(s)
    timing_stack = []


def write_file(filename, code):
    with open(filename, "w") as f:
        f.write(code)


def shell(cmd, verbose=False):
    if verbose:
        print(cmd)
    res = subprocess.check_output(cmd, shell=True)
    if verbose:
        print(res)


def test_kernel(kernel_name, config, do_mem_mapping=True):

    @timing
    def generate(kernel_name, config):
        # Generate C code
        kernel = Kernel(kernel_name, config)
        kernel.process(do_mem_mapping=do_mem_mapping)

        hname, hcode = kernel.generate_header()
        if do_mem_mapping:
            log.debug(c_highlight(hcode))
        cname, ccode = kernel.generate_benchmark()
        # print(c_highlight(ccode))
        gdbname, gdbcode, dumpfiles = kernel.generate_benchmark_gdb(
            prefix=PREFIX + ("dma_" if do_mem_mapping else "")
        )
        return (hname, hcode, cname, ccode, gdbname, gdbcode), dumpfiles

    @timing
    def write_files(hname, hcode, cname, ccode, gdbname, gdbcode):
        hname = PREFIX + hname
        cname = PREFIX + cname
        gdbname = PREFIX + gdbname
        write_file(hname, hcode)
        write_file(cname, ccode)
        write_file(gdbname, gdbcode)
        return hname, cname, gdbname
    
    @timing
    def build(cnames, binname):
        cmd = f"{CC} {CFLAGS} {LDFLAGS} {' '.join(cnames)} -I{PREFIX} -o {binname}"
        ret = shell(cmd)
        return ret
    
    @timing
    def run_simu(gdbname, binname):
        # Run
        cmd = f"gdb --batch --command={gdbname} --args {binname}"
        ret = shell(cmd)
        shell(f'wc -c {" ".join(dumpfiles)}')
        return dumpfiles
    
    binname = PREFIX + "sma_bin" + ("_mem_mapping" if do_mem_mapping else "")
    files_plus_code, dumpfiles = generate(kernel_name, config)
    hname, cname, gdbname = write_files(*files_plus_code)
    build([cname], binname)
    run_simu(gdbname, binname)
    return dumpfiles



def validation_kernel(kernel_name, config):
    log.info(f"KERNEL validation: {kernel_name} {config}")
    files = [
        test_kernel(kernel_name, config, do_mem_mapping=test) for test in (False, True)
    ]
    total_diff = 0
    for ff in zip(*tuple(files)):
        eq = filecmp.cmp(*ff)   
        total_diff += not eq
        log.debug(f'{ff[0]} and {ff[1]} {"are equals" if eq else "differ"}')

    return total_diff




from itertools import product

CONF_1D_04 = (2, 7, 8, 31, 32, 111, 1000, 1024)
CONF_1D_CONF_11 = (2, 4, 8, 10, 20, 50, 256, 400, 1024, 2000, 2048)

CONF_2D_04_04 =  product(CONF_1D_04, CONF_1D_04)
M_N_BENCHMARK = [{'M': m, 'N': n} for m, n in CONF_2D_04_04]


BIG_BENCHMARK = {name: M_N_BENCHMARK for name in ("set1", "copy", "gemv")}



RNG_1D = (5, 8, 32, 64, 255, 256)
CFG_X_Y_DKX_DKY = [{'X': x, 'Y': y, 'DKX': dkx, 'DKY': dky} for x, y, dkx, dky 
                    in product(RNG_1D, RNG_1D, (1, 3), (3, 4))]


def gen_bench(bench):
    for name, configs in bench.items():
        for config in configs:
            yield name, config


def ddt_bench(bench):
    class WrapList(list):
        pass

    def annotated(kv):
        r = WrapList([*kv])
        setattr(r, "__name__", kernel_compute_name(*kv))
        return r
        
    return tuple(map(annotated, bench))


@ddt
class TestKernels(unittest.TestCase):
    @data(*ddt_bench(gen_bench(BIG_BENCHMARK)))
    def test_all(self, args):
        self.assertFalse(validation_kernel(*args))

    @data(*ddt_bench(gen_bench({"conv2d": CFG_X_Y_DKX_DKY})))
    def test_conv2d(self, args):
        self.assertFalse(validation_kernel(*args))


if __name__ == '__main__':
    # validation_kernel('conv2d', {'X': 256, 'Y': 8, 'DKX': 3, 'DKY': 3})
    # validation_kernel('conv2d', {'X': 32, 'Y': 8, 'DKX': 1, 'DKY': 4})
    # validation_kernel('set1', {'M': 200, 'N': 2})
    # X64_Y5_DKX1_DKY4
    validation_kernel("set10", {'N': 32})
    timing_dump()