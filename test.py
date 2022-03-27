
import logging as log
import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.WARNING
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
CFLAGS = "-Wall -Wextra -Werror -Idmasimulator -g"
LDFLAGS = ""


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
    # print(gdbcode)

    hname = PREFIX + hname
    cname = PREFIX + cname
    gdbname = PREFIX + gdbname
    binname = PREFIX + "sma_bin" + ("_mem_mapping" if do_mem_mapping else "")
    write_file(hname, hcode)
    write_file(cname, ccode)
    write_file(gdbname, gdbcode)

    cmd = f"{CC} {CFLAGS} {LDFLAGS} {cname} -I{PREFIX} -o {binname}"
    ret = shell(cmd)
    # Run
    cmd = f"gdb --batch --command={gdbname} --args {binname}"
    ret = shell(cmd)
    shell(f'wc -c {" ".join(dumpfiles)}')
    return dumpfiles
    # dump binary memory file.bin weights (char*)weights + sizeof(weights)


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



class WrapList(list):
    pass

def annotated(name, config):
    r = WrapList([name, config])
    setattr(r, "__name__", kernel_compute_name(name, config))
    return r




BIG_BENCHMARK = {"gemv": [{'M': m, 'N': n} for m, n in [(2, 2),
                                                        (2, 64),
                                                        (64, 2),
                                                        (64, 64),
                                                        (31, 13),
                                                        (45, 12),
                                                        (2, 34),
                                                        (12, 4),
                                                        (111, 111),
                                                        (16, 128)]]}

def gen_bench(bench):
    for name, configs in BIG_BENCHMARK.items():
        for config in configs:
            yield name, config

bench = (annotated(k, v) for k, v in gen_bench(BIG_BENCHMARK))



@ddt
class TestKernels(unittest.TestCase):
    def test_small_config(self):
        self.assertFalse(validation_kernel("gemv", {'M': 16, 'N': 129*2}))

    @data(*tuple(bench))
    def test_all(self, args):
        self.assertFalse(validation_kernel(*args))



suite = unittest.TestLoader().loadTestsFromTestCase(TestKernels)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)
