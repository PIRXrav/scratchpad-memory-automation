#pragma once

#define DMA_SIZE 32

#define DMA_LD(adr, buff, size) memcpy(buff, adr, size)
#define DMA_ST(adr, buff, size) memcpy(adr, buff, size)
