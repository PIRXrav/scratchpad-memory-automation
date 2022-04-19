#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


/******************************* Software part ********************************/

#define DMA_SIZE 128
#define NR_DMA 3
#define WORD_SIZE 8

#define DMA_INIT(index, adr, size) __sma_dma_init(index, adr, size)
#define DMA_LD(index) __sma_dma_load(index)
#define DMA_ST(index) __sma_dma_store(index)
#define DMA_RW(index, reladr)  __sma_dma_access(index, reladr)
#define MIN(a, b) ((a) < (b) ? (a) : (b)) 

/******************************* Hardware part ********************************/

#include <assert.h>


//#define HW_ALIGN_CONSTRAINTS
#ifdef HW_ALIGN_CONSTRAINTS

#define __SMA_RAM_PTR
#define __SMA_RAM __attribute__((aligned(WORD_SIZE)))

#define SMA_ALIGNMENT WORD_SIZE
#define SMA_MULTIPLICITY WORD_SIZE

#define __DMA_EFF_SIZE (DMA_SIZE + SMA_MULTIPLICITY)
 
typedef char __sma__dma_t;
__sma__dma_t __sma_dma[NR_DMA][__DMA_EFF_SIZE];
uint8_t __sma_offset[NR_DMA];
uint16_t __sma_size[NR_DMA];
__SMA_RAM_PTR void * __sma_base_adr[NR_DMA];

/**
 * 8bytes align & 8bytes multipicity
 */ 
void __sma_dma_init(uint8_t index, __SMA_RAM_PTR void *adr, uint16_t size){
    // Alignment
    __sma_offset[index] = ((uint64_t)adr) % SMA_ALIGNMENT;
    __sma_base_adr[index] = adr - __sma_offset[index];
    assert((uint64_t)__sma_base_adr[index] % SMA_ALIGNMENT == 0);
    size += __sma_offset[index];
    // Multiplicity
    __sma_size[index] = size + ((SMA_MULTIPLICITY - (size % SMA_MULTIPLICITY)) % SMA_MULTIPLICITY);
    assert(__sma_size[index] % SMA_MULTIPLICITY == 0);
    assert(__sma_size[index] < __DMA_EFF_SIZE);
    // printf("Initialise DMA %d @ %p->%p # %d->%d\n", index, adr, __sma_base_adr[index], size, __sma_size[index]);
}

void __sma_dma_load(uint8_t index){
    assert(__sma_size[index] <= __DMA_EFF_SIZE);
    assert(__sma_size[index] % SMA_MULTIPLICITY == 0);
    assert((uint64_t)__sma_base_adr[index] % SMA_ALIGNMENT == 0);
    memcpy(&__sma_dma[index][0], __sma_base_adr[index], __sma_size[index]);
}

void __sma_dma_store(uint8_t index){
    assert(__sma_size[index] <= __DMA_EFF_SIZE);
    assert(__sma_size[index] % SMA_MULTIPLICITY == 0);
    assert((uint64_t)__sma_base_adr[index] % SMA_ALIGNMENT == 0);
    memcpy(__sma_base_adr[index], &__sma_dma[index][0], __sma_size[index]);
}

void *__sma_dma_access(uint8_t index, uint16_t reladr){
    assert(reladr < DMA_SIZE);
    assert((reladr + __sma_offset[index]) < __DMA_EFF_SIZE);
    return &__sma_dma[index][reladr + __sma_offset[index]];
}

#else

#define __SMA_RAM_PTR
#define __SMA_RAM __attribute__((aligned(8)))

typedef char __sma__dma_t;
__sma__dma_t __sma_dma[NR_DMA][DMA_SIZE];
uint16_t __sma_size[NR_DMA];
__SMA_RAM_PTR void * __sma_base_adr[NR_DMA];

void __sma_dma_init(uint8_t index, __SMA_RAM_PTR void *adr, uint16_t size){
    __sma_base_adr[index] = adr;
    __sma_size[index] = size;
    
    if(!(__sma_size[index] <= DMA_SIZE)){
        exit(1);
    }
    /*
    if(!(__sma_size[index] % WORD_SIZE == 0)){
        exit(2);
    }
    if(!((uint64_t)__sma_base_adr[index] % WORD_SIZE == 0)){
        exit(10);
    }*/
}

void __sma_dma_load(uint8_t index){
    memcpy(&__sma_dma[index][0], __sma_base_adr[index], __sma_size[index]);
}

void __sma_dma_store(uint8_t index){
    memcpy(__sma_base_adr[index], &__sma_dma[index][0], __sma_size[index]);
}

void *__sma_dma_access(uint8_t index, uint16_t reladr){
    if(!(reladr < DMA_SIZE)){
        exit(20);
    }
    return &__sma_dma[index][reladr];
}


#endif