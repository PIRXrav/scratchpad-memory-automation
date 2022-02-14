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
