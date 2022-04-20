"""Python codegen for using dma.h
"""

class Gencode:
    @classmethod
    def cgen_dma_init(self, index, adr, size):
        return f"DMA_INIT({index}, {adr}, {size});"

    @classmethod
    def cgen_dma_ld(self, index, adr, size):
        return f"DMA_LD({index})"

    @classmethod
    def cgen_dma_st(self, index, adr, size):
        return f"DMA_ST({index})"

    @classmethod
    def cgen_static_mac(self, A, B):
        return "+".join(f"(({a}) * ({b}))" for a, b in zip(A, B))


def fix_size_to_word_size(size, word_size):
    return size + ((word_size - (size % word_size)) % word_size)
