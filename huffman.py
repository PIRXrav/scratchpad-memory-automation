from heapq import heappush, heappop, heapify
from collections import Counter


class Huffman:
    def __init__(self, txt):
        # Compute symbol frequencies
        self.symb2freq = Counter(txt)
        # Create symbol table
        heap = [[wt, [sym, ""]] for sym, wt in self.symb2freq.items()]
        heapify(heap)
        while len(heap) > 1:
            lo = heappop(heap)
            hi = heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        self.huff = sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

    def encode(self, txt):
        pass

    def __str__(self):
        return f'{self.symb2freq=}\n{self.huff=}'


if __name__ == '__main__':
    txt = "aaaaaabcd"
    huff = Huffman(txt)
    print(huff)
