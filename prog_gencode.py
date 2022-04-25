"""Prog code generator
"""

from prog import ProgVisitor
from itertools import chain
import struct

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


class Frame:
    def __init__(self, dma):
        self.PAYLOAD_SIZE = 16  # Byte
        self.TYPE_SIZE = 2  # Byte
        self.MAX_MV = (dma - self.PAYLOAD_SIZE) // self.TYPE_SIZE
        self.addr_i = -1  # Default value
        self.size_i = 0
        self.addr_o = -1  # Default value
        self.size_o = 0
        self.addr_store = -1  # Default value
        self.size_store = 0
        self.nb_moves = 0  # Must be incremented
        self.next_size = None  # Must be defined in frame chain
        self.rawa = []  # All RW dma addrs

    def append_rw(self, ra, wa):
        self.rawa.append(ra)
        self.rawa.append(wa)
        self.nb_moves += 1

    def is_filled(self):
        return len(self.rawa) + 2 > self.MAX_MV

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        arg = (f'{p}(@{a}#{s})' for p, a, s in (('ldi', self.addr_i, self.size_i),
                                                ('ldo', self.addr_o, self.size_o),
                                                ('sto', self.addr_store, self.size_store)))
        return f'Frame({",".join(arg)},RAWA#{len(self.rawa)}x{self.TYPE_SIZE})'

    def chain(self, word_size, next_size):
        self.next_size = next_size
        base_size = 16 * 2 + len(self.rawa) * 2
        paddind = (word_size - (base_size % word_size)) % word_size
        fmt = 'H' * 8 + 'H' * len(self.rawa) + 'x' * paddind
        payload = [self.addr_i, self.size_i, self.addr_o, self.size_o,
                   self.addr_store, self.size_store, self.nb_moves, self.next_size]
        data = self.rawa
        pad = [0] * paddind
        print(payload, data, pad)
        self.raw = struct.pack(fmt, *chain(payload, data, pad))
        return len(self.raw)

    def as_array(self):
        """Return self as serial format (memory image)
        """
        return self.raw


class FrameChain:
    """Chain of frame to ensure coherence between all frames
    """
    def __init__(self, dma, word_size):
        self.dma = dma
        self.word_size = word_size
        self._head = Frame(self.dma)  # current frame
        self._frames = []

    def _push(self):
        self._frames.append(self._head)  # Push head frame
        self.head = Frame(self.dma)  # Create new frame

    def ldi(self, addr, size):
        if self._head.addr_i != -1:  # Push frame if needed
            self._push()
        self._head.addr_i = addr
        self._head.size_i = size

    def ldo(self, addr, size):
        if self._head.addr_o != -1:  # Push frame if needed
            self._push()
        self._head.addr_o = addr
        self._head.size_o = size

    def sto(self, addr, size):
        self._head.addr_store = addr
        self._head.size_store = size
        self._push()  # We must push frame

    def mv(self, ra, wa):
        self._head.append_rw(ra, wa)
        if self._head.is_filled():  # Push if filled
            self._push()

    def as_array(self, word_size):
        # Chain all frames
        next_size = self._frames[-1].chain(word_size, 0)
        for f in reversed(self._frames[:-1]):
            next_size = f.chain(word_size, next_size)

        # Export
        return next_size, b''.join(f.as_array() for f in self._frames)

    def __str__(self):
        return str(self._frames)


class GenericGencodeProgVisitor(ProgVisitor):
    """
    GenericGencodeProgVisitor

    struct  payload{

    }

    BASE_ADDR = XXXX
    BASE_SIZE = XXXX

    while(BASE_SIZE){
        // Update next frame addr
        BASE_ADDR += BASE_SIZE

        // Read current frame
        DMA_LD_PROG @ BASE_ADDR # BASE_SIZE -> read

        // Analyse Payload #16 bytes
        addr_i = read[0] // 16b
        size_i = read[1] // 16b
        addr_o = read[2] // 16b
        size_o = read[3] // 16b
        addr_store = read[4] // 16b. We use replication of dma_o addr to avoid state in code
        size_store = read[5] // 16b
        nb_moves = read[6] // 16b
        BASE_SIZE = read[7] // 16b. if O: EOS

        // Process frame
        base_rw = read + sizeof(payload);

        if(addr_i != -1){
            DMA_LDI @ addr_i # size_i -> dmai
        }
        if(addr_o != -1){
            DMA_LDO @ addr_o # size_o -> dmao
        }
        for(size_t i= 0, i < nb_moves; i++){
            base_rw[i*2], base_rw[8 + i*2 + 1]
            COPY
        }
        if(addr_store != -1){
            DMA_STO @ addr_store # size_store <- dmao
        }
    }
    """
    def __init__(self, prog, dma, word_size):
        self.dma = dma
        self.fc = FrameChain(dma, word_size)
        self.visit(prog)

    def visit_ldi(self, rel_addr, size):
        self.fc.ldi(rel_addr, size)

    def visit_ldo(self, rel_addr, size):
        self.fc.ldo(rel_addr, size)

    def visit_sto(self, rel_addr, size):
        self.fc.sto(rel_addr, size)

    def visit_mv(self, rel_addr_dst, rel_addr_src):
        self.fc.mv(rel_addr_dst, rel_addr_src)

    def export(self):
        return self.fc
