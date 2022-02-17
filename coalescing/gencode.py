from prog import ProgVisitor

class CLinearGencodeProgVisitor(ProgVisitor):
    def __init__(self, prog, iname, oname, dmai, dmao, dma):
        self.res = []
        self.iname = iname
        self.oname = oname
        self.dmai = dmai
        self.dmao = dmao
        self.dma = dma
        self.visit(prog)
    
    def visit_ldi(self, rel_addr):
        self.res.append(f'DMA_LD({self.iname}+{rel_addr}, {self.dmai}, {self.dma});')

    def visit_ldo(self, rel_addr):
        self.res.append(f'DMA_LD({self.oname}+{rel_addr}, {self.dmao}, {self.dma});')

    def visit_sto(self, rel_addr):
        self.res.append(f'DMA_ST({self.oname}+{rel_addr}, {self.dmao}, {self.dma});')

    def visit_mv(self, rel_addr_dst, rel_addr_src):
        self.res.append(f'*({self.dmao}+{rel_addr_dst}) = *({self.dmai}+{rel_addr_src});')

    def export(self):
        return "\n".join(self.res)
    
    def __str__(self):
        return self.export()


class CMvTabGencodeProgVisitor(ProgVisitor):
    """
    ST
    LD
    for(; i < XX; i++){
        *(dmao + otab[i]) = ..
    }
    """
    def __init__(self, prog, iname, oname, dmai, dmao, dma):
        self.itab = []
        self.otab = []
        self.cpt = 0
        self.mv_cpt = 0

        self.res = []
        self.iname = iname
        self.oname = oname
        self.dmai = dmai
        self.dmao = dmao
        self.dma = dma
        self.visit(prog)

    def append_loop(self):
        if self.cpt != 0:
            self.res.append(f'for(; mv_cpt < {self.mv_cpt}; mv_cpt++){{*({self.dmao}+otab[mv_cpt]) = *({self.dmai}+itab[mv_cpt]);}}')
            self.cpt = 0
            self.in_for = False

    def visit_ldi(self, rel_addr):
        self.append_loop()
        self.res.append(f'DMA_LD({self.iname}+{rel_addr}, {self.dmai}, {self.dma});')

    def visit_ldo(self, rel_addr):
        self.append_loop()
        self.res.append(f'DMA_LD({self.oname}+{rel_addr}, {self.dmao}, {self.dma});')

    def visit_sto(self, rel_addr):
        self.append_loop()
        self.res.append(f'DMA_ST({self.oname}+{rel_addr}, {self.dmao}, {self.dma});')

    def visit_mv(self, rel_addr_dst, rel_addr_src):
        self.cpt += 1
        self.mv_cpt += 1
        self.itab.append(rel_addr_src)
        self.otab.append(rel_addr_dst)

    def export(self):
        self.citab = f'const int16_t itab[{self.mv_cpt}] = {{{", ".join(map(str, self.itab))}}};\n'
        self.cotab = f'const int16_t otab[{self.mv_cpt}] = {{{", ".join(map(str, self.otab))}}};\n'
        self.coreloop = 'int32_t mv_cpt = 0;\n' +  '\n'.join(self.res)
        return self.citab, self.cotab, self.coreloop
    
    def __str__(self):
        return "__str__"
