
import logging as log
import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.INFO
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(filename)-16s:%(lineno)-4d>> %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

log = logging.getLogger()
from isort import file
import subprocess
from kernelgenerator import Kernel
import filecmp
from asttools import c_highlight
import sys

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
    # print(c_highlight(hcode))
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


BIG_BENCHMARK = {"gemv": [{'M': m, 'N': n} for m, n in [(1, 1),(1, 64),(64, 1),(64, 61)]]}

if __name__ == "__main__":
    for name, configs in BIG_BENCHMARK.items():
        log.info(f'Benchmark : {name} # {len(configs)}')
        for config in configs:
            res = validation_kernel(name, config)
            if res:
                raise Exception("Result differ")
            else:
                log.info('Test passed')
    
    exit(res)
