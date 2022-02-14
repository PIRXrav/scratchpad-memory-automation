
class Prog:
    LDI = 'ldi'
    LDO = 'ldo'
    STO = 'sto'
    MV = 'mv'

    def __init__(self):
        self.prog = []
    
    def append_ldi(self, rel_addr):
        self.prog.append((self.LDI, rel_addr))
    
    def append_ldo(self, rel_addr):
        self.prog.append((self.LDO, rel_addr))
    
    def append_sto(self, rel_addr):
        self.prog.append((self.STO, rel_addr))
    
    def append_mv(self, rel_addr_dst, rel_addr_src):
        self.prog.append((self.MV, rel_addr_dst, rel_addr_src))
    
    def evaluate(self, dma):
        count_total = len(self.prog)
        count_ldi = len(list(filter(lambda x:x[0] == 'ldi', self.prog)))
        count_ldo = len(list(filter(lambda x:x[0] == 'ldo', self.prog)))
        count_sto = len(list(filter(lambda x:x[0] == 'sto', self.prog)))
        count_mv = len(list(filter(lambda x:x[0] == 'mv', self.prog)))

        res = {'total inst': count_total,
            'total ldi': count_ldi,
            'total ldo': count_ldo,
            'total mv': count_mv,
            'total ldst': count_ldi + count_ldo + count_sto,
            'DMA': dma,
            'quality ldi': count_mv / count_ldi / dma,
            'quality sto': count_mv / count_sto / dma,
            'quality global': count_mv / (count_ldi + count_ldo + count_sto) / dma * 3}
        print(f'=========== EVALUATE PROG ===========')
        print('\n'.join((f'{k:>20}; {v}' for k, v in res.items())))
        return res


class ProgVisitor:
    """
    Using a visitor pattern for different gencode
    """
    def visit(self, prog, *args, **kwargs):
        for instr in prog.prog:
            if instr[0] == Prog.LDI:
                self.visit_ldi(instr[1], *args, *kwargs)
            elif instr[0] == Prog.LDO:
                self.visit_ldo(instr[1], *args, *kwargs)
            elif instr[0] == Prog.STO:
                self.visit_sto(instr[1], *args, *kwargs)
            elif instr[0] == Prog.MV:
                self.visit_mv(instr[1], instr[2], *args, *kwargs)
                
    def visit_ldi(self, rel_addr, *args, **kwargs):
        raise Exception('abstract method')

    def visit_ldo(self, rel_addr, *args, **kwargs):
        raise Exception('abstract method')

    def visit_sto(self, rel_addr, *args, **kwargs):
        raise Exception('abstract method')

    def visit_mv(self, rel_addr_dst, rel_addr_src, *args, **kwargs):
        raise Exception('abstract method')