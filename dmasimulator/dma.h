#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DMA_SIZE 32

#define DMA_LD(adr, buff, size) memcpy(buff, adr, size)
#define DMA_ST(adr, buff, size) memcpy(adr, buff, size)

#define MIN(a, b) ((a) < (b) ? (a) : (b)) 

uint8_t __SMA__dma0[DMA_SIZE];
uint8_t __SMA__dma1[DMA_SIZE];
uint8_t __SMA__dma2[DMA_SIZE];
uint8_t __SMA__dma3[DMA_SIZE];
