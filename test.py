"""SMA unit tests
"""

import logging
from kernelgenerator import Kernel, kernel_compute_name
import filecmp
from asttools import c_highlight
import unittest
from ddt import ddt, data
from itertools import product

import toolchain as tc

DEBUG_MODE = 1

if __name__ == "__main__":
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.ERROR
    DEBUG_MODE = 0

LOGFORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s %(filename)-16s:%(lineno)-4d>> %(message)s",
    handlers=[logging.StreamHandler()],
)

log = logging.getLogger()


def validation_kernel(kernel_name, config):
    """Run kernel with and without dma mapping. Then compare the memory of the
    programmes with gdb
    """
    log.info(f"KERNEL validation: {kernel_name} {config}")
    kernel = Kernel(kernel_name, config)
    result, files = kernel.bench()
    print(files)
    # for ff in zip(*tuple(files)):
    #     eq = filecmp.cmp(*ff)
    #     total_diff += not eq
    #     log.debug(f'{ff[0]} and {ff[1]} {"are equals" if eq else "differ"}')

    return result


CONF_1D_04 = (2, 7, 8, 31, 32, 111, 1000, 1024)
CONF_1D_CONF_11 = (2, 4, 8, 10, 20, 50, 256, 400, 1024, 2000, 2048)

CONF_2D_04_04 = product(CONF_1D_04, CONF_1D_04)
M_N_BENCHMARK = [{"M": m, "N": n} for m, n in CONF_2D_04_04]


BIG_BENCHMARK = {name: M_N_BENCHMARK for name in ("set1", "copy", "gemv")}


RNG_1D = (5, 8, 32, 64, 255, 256)
CFG_X_Y_DKX_DKY = [
    {"X": x, "Y": y, "DKX": dkx, "DKY": dky}
    for x, y, dkx, dky in product(RNG_1D, RNG_1D, (1, 3), (3, 4))
]

CFG_N = [{"N": n} for n in [1, 2, 3, 4, 8, 9, 20, 32, 128, 255, 1010, 4096]]


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


FAST_BENCH = {
    "conv2d": [
        {"X": 256, "Y": 8, "DKX": 3, "DKY": 3},
        {"X": 32, "Y": 8, "DKX": 1, "DKY": 4},
    ],
    "copy": [{"M": 32, "N": 1024}],
    "gemv": [{"M": 32, "N": 32}],
    "set1_32b": [{"N": 128}],
    "set10": [{"N": 32}],
}


@ddt
class TestKernels(unittest.TestCase):
    @data(*ddt_bench(gen_bench(FAST_BENCH)))
    def test_0_fast(self, args):
        self.assertFalse(validation_kernel(*args))

    @data(*ddt_bench(gen_bench(BIG_BENCHMARK)))
    def test_1_base(self, args):
        self.assertFalse(validation_kernel(*args))

    @data(*ddt_bench(gen_bench({"set1_32b": CFG_N})))
    def test_2_multiple_types(self, args):
        self.assertFalse(validation_kernel(*args))

    @data(*ddt_bench(gen_bench({"conv2d": CFG_X_Y_DKX_DKY})))
    def test_conv2d(self, args):
        self.assertFalse(validation_kernel(*args))


if __name__ == "__main__":
    # validation_kernel()
    # validation_kernel('conv2d', )
    # validation_kernel('copy', )
    validation_kernel('conv2d', {"X": 256, "Y": 8, "DKX": 3, "DKY": 3})
    # validation_kernel('conv2d', {"X": 32, "Y": 8, "DKX": 1, "DKY": 4})
    # validation_kernel('gemv', {"M": 32, "N": 32})

    # validation_kernel("set1_32b", {"N": 128})
    # X64_Y5_DKX1_DKY4
    # validation_kernel("set10", {"N": 32})
