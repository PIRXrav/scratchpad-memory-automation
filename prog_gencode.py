"""Prog code generator
"""

from prog import ProgVisitor

# OUTDATED
# class CLinearGencodeProgVisitor(ProgVisitor):
#     def __init__(self, prog, iname, oname, dmai, dmao, dma):
#         self.res = []
#         self.iname = iname
#         self.oname = oname
#         self.dmai = dmai
#         self.dmao = dmao
#         self.dma = dma
#         self.visit(prog)
#
#     def visit_ldi(self, rel_addr, size):
#         self.res.append(f'DMA_LD({self.iname}+{rel_addr}, {self.dmai}, {self.dma});')
#
#     def visit_ldo(self, rel_addr, size):
#         self.res.append(f'DMA_LD({self.oname}+{rel_addr}, {self.dmao}, {self.dma});')
#
#     def visit_sto(self, rel_addr, size):
#         self.res.append(f'DMA_ST({self.oname}+{rel_addr}, {self.dmao}, {self.dma});')
#
#     def visit_mv(self, rel_addr_dst, rel_addr_src):
#         self.res.append(f'*({self.dmao}+{rel_addr_dst}) = *({self.dmai}+{rel_addr_src});')
#
#     def export(self):
#         return "\n".join(self.res)
#
#     def __str__(self):
#         return self.export()

from gencode_dma import Gencode

class CMvTabGencodeProgVisitor(ProgVisitor):
    """
    ST
    LD
    for(; i < XX; i++){
        *(dmao + otab[i]) = ..
    }
    """
    def __init__(self, prog, iname, oname, dma, dtype):
        self.itab = []
        self.otab = []
        self.cpt = 0
        self.mv_cpt = 0

        self.res = []
        self.iname = iname
        self.oname = oname
        self.dma = dma
        self.dtype = dtype

        self.i_cfg = None
        self.o_cfg = None

        self.visit(prog)

    def append_loop(self):
        if self.cpt != 0:
            dest = f"*(({self.dtype}*)DMA_RW(1, otab[mv_cpt]))"
            srce = f"*(({self.dtype}*)DMA_RW(0, itab[mv_cpt]))"
            assign = f"{dest} = {srce};"
            self.res.append(f"for(; mv_cpt < {self.mv_cpt}; mv_cpt++){{{assign}}}")
            self.cpt = 0
            self.in_for = False

    def visit_ldi(self, rel_addr, size):
        self.append_loop()
        self.i_cfg = ('0', f'(char*){self.iname}+{rel_addr}', f'{size}')  # DMA 0 for I
        self.res.append(Gencode.cgen_dma_init(*self.i_cfg) + ';')
        self.res.append(Gencode.cgen_dma_ld(*self.i_cfg) + ';')

    def visit_ldo(self, rel_addr, size):
        self.append_loop()
        self.o_cfg = ('1', f'(char*){self.oname}+{rel_addr}', f'{size}')  # DMA 1 for O
        self.res.append(Gencode.cgen_dma_init(*self.o_cfg) + ';')
        self.res.append(Gencode.cgen_dma_ld(*self.o_cfg) + ';')

    def visit_sto(self, rel_addr, size):
        # Assume ever sto after ldo
        self.append_loop()
        self.res.append(Gencode.cgen_dma_st(*self.o_cfg) + ';')

    def visit_mv(self, rel_addr_dst, rel_addr_src):
        self.cpt += 1
        self.mv_cpt += 1
        self.itab.append(rel_addr_src)
        self.otab.append(rel_addr_dst)

    def export(self):
        self.citab = f'const int16_t itab[{self.mv_cpt}] = {{{", ".join(map(str, self.itab))}}};\n'
        self.cotab = f'const int16_t otab[{self.mv_cpt}] = {{{", ".join(map(str, self.otab))}}};\n'
        self.coreloop = 'int32_t mv_cpt = 0;\n' + '\n'.join(self.res)
        return self.citab, self.cotab, self.coreloop

    def __str__(self):
        return "__str__"


# TODO GenericGencodeProgVisitor
# for X in range (XXX){
#   BASE_ADDR_I[XXX] = .
#   SIZE_I[XXX]      = .
#   BASE_ADDR_O[XXX] = .
#   SIZE_O[XXX]      = .
#   BASE_REPEAT[XXX] = .
#   DMA_INIT(0, (char *)tab0 + 64, 128);
#   DMA_INIT(1, (char *)tab1 + 64, 128);
#   DMA_INIT(2, (char *)_SMA_INDIRECTION + 64, 128);
#
#   if(need_update0):
#     DMA_LD(0);
#   if(need_update0):
#     DMA_LD(1);
#
#   for j in BASE_REPEAT[XXX]{
#     REL_CPT = REL[XXX][j]
#     DMA_LD(2)
#     for (i = 0; i < REL_CPT; i++) {
#       *((int8_t *)DMA_RW(1, DMA_RW(2, i*2))) = *((int8_t *)DMA_RW(0, DMA_RW(2, i*2+1)));
#     }
#   }
#   if(need_update1):
#     DMA_ST(1);
# }
