import numpy as np

class Prog:
    LDI = 'ldi'
    LDO = 'ldo'
    STO = 'sto'
    MV = 'mv'

    def __init__(self):
        self.prog = []

    def append_ldi(self, rel_addr, size):
        self.prog.append((self.LDI, rel_addr, size))

    def append_ldo(self, rel_addr, size):
        self.prog.append((self.LDO, rel_addr, size))

    def append_sto(self, rel_addr, size):
        self.prog.append((self.STO, rel_addr, size))

    def append_mv(self, rel_addr_dst, rel_addr_src):
        self.prog.append((self.MV, rel_addr_dst, rel_addr_src))

    def gen_evaluation(self, dma):
        count_total = len(self.prog)
        count_ldi = len(list(filter(lambda x: x[0] == 'ldi', self.prog)))
        count_ldo = len(list(filter(lambda x: x[0] == 'ldo', self.prog)))
        count_sto = len(list(filter(lambda x: x[0] == 'sto', self.prog)))
        count_mv = len(list(filter(lambda x: x[0] == 'mv', self.prog)))
        res = {'total inst': count_total,
               'total ldi': count_ldi,
               'total ldo': count_ldo,
               'total mv': count_mv,
               'total ldst': count_ldi + count_ldo + count_sto,
               'DMA': dma,
               'quality ldi': count_mv / count_ldi / dma,
               'quality sto': count_mv / count_sto / dma,
               'quality global': count_mv / (count_ldi + count_ldo + count_sto) / dma * 3}
        print('=========== EVALUATE PROG ===========')
        print('\n'.join((f'{k:>20}; {v}' for k, v in res.items())))
        return res

    def gen_hist_dma_repartition(self, xsize=20, ysize=20):
        """Generate ascii histogram of DMA rd/wr
        """
        class AnalyserDmaValueRepartitionProgVisitor(ProgVisitor):
            """Analyse DMA value repartition
            """
            def __init__(self, prog):
                self.iaddrs = []
                self.oaddrs = []
                self.visit(prog)
                self.datas = (self.iaddrs, self.oaddrs)
                self.names = ('dma i addrs', 'dma o addrs')

            def visit_ldi(self, rel_addr, size, *args, **kwargs):
                pass

            def visit_ldo(self, rel_addr, size, *args, **kwargs):
                pass

            def visit_sto(self, rel_addr, size, *args, **kwargs):
                pass

            def visit_mv(self, rel_addr_dst, rel_addr_src, *args, **kwargs):
                self.iaddrs.append(rel_addr_src)
                self.oaddrs.append(rel_addr_dst)

        pv = AnalyserDmaValueRepartitionProgVisitor(self)
        from termhist import termhists
        return termhists(pv.datas, pv.names, xsize=xsize, ysize=ysize)
    
    def gen_statistics_dma_adr(self):
        class MvCollectAddrsProgVisitor(ProgVisitor):
            """."""
            def __init__(self, prog):
                self.itab = []
                self.otab = []
                self.visit(prog)

            def visit_mv(self, rel_addr_dst, rel_addr_src):
                self.itab.append(rel_addr_src)
                self.otab.append(rel_addr_dst)

            def export(self):
                return np.array(self.itab, dtype=np.int16), np.array(self.otab, dtype=np.int16)

        itab, otab = MvCollectAddrsProgVisitor(self).export()
        from numpy import asarray
        from numpy import savetxt
        savetxt('itab', itab)
        savetxt('otab', otab)
        

class ProgVisitor:
    """Using a visitor pattern for different gencode
    """
    def visit(self, prog, *args, **kwargs):
        for instr in prog.prog:
            if instr[0] == Prog.LDI:
                self.visit_ldi(instr[1], instr[2], *args, *kwargs)
            elif instr[0] == Prog.LDO:
                self.visit_ldo(instr[1], instr[2], *args, *kwargs)
            elif instr[0] == Prog.STO:
                self.visit_sto(instr[1], instr[2], *args, *kwargs)
            elif instr[0] == Prog.MV:
                self.visit_mv(instr[1], instr[2], *args, *kwargs)

    def visit_ldi(self, rel_addr, size, *args, **kwargs):
        pass

    def visit_ldo(self, rel_addr, size, *args, **kwargs):
        pass

    def visit_sto(self, rel_addr, size, *args, **kwargs):
        pass

    def visit_mv(self, rel_addr_dst, rel_addr_src, *args, **kwargs):
        pass

if __name__ == '__main__':
    from test_coalescing import Coalescing
    from optimizer import toeplitz, export
    import numpy as np

    # DMA config
    WORD_SIZE = 8
    DMA = 128  # 1024o
    # Input
    x = 16
    y = 16
    # Filter shape
    Dkx = 2
    Dky = 2
    DTYPE = "int8_t"
    tensor_i = np.arange(x * y, dtype=np.int32).reshape(y, x)  # tab[index] = index !!
    tensor_o = toeplitz(tensor_i, y, x, Dky, Dkx)

    prog = Coalescing(tensor_i, tensor_o, 'TODO:TYPESIZE', DMA, WORD_SIZE).export()
    r = prog.gen_statistics_dma_adr()
