#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


#define __SMA_RAM_PTR
#define __SMA_RAM __attribute__((aligned(8)))

void __sma_dma_init(uint8_t index, __SMA_RAM_PTR void *adr, uint16_t size);
void __sma_dma_load(uint8_t index);
void __sma_dma_store(uint8_t index);
void *__sma_dma_access(uint8_t index, uint16_t reladr);


/******************************* Software part ********************************/

#ifndef DMA_SIZE
#error DMA_SIZE undefined
#endif
#ifndef WORD_SIZE
#error WORD_SIZE undefined
#endif
#define NR_DMA 3

#define DMA_INIT(index, adr, size) __sma_dma_init(index, adr, size)
#define DMA_LD(index) __sma_dma_load(index)
#define DMA_ST(index) __sma_dma_store(index)
#define DMA_RW(index, reladr)  __sma_dma_access(index, reladr)
#define MIN(a, b) ((a) < (b) ? (a) : (b)) 

