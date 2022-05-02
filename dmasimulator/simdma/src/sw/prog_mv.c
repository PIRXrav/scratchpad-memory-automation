#include <stdint.h>
#include <stdio.h>
#include "dma.h"
#include "prog_mv.h"

struct prog_header_t{
    int16_t addr_i;
    uint16_t size_i;
    int16_t addr_o;
    uint16_t size_o;
    int16_t addr_store; // 16b. We use replication of dma_o addr to avoid state in code
    uint16_t size_store;
    uint16_t nb_moves;
    uint16_t base_size; //  if O: EOS
};

/*
Mapping:
    DMA 0: tensor_i
    DMA 1: tensor_o
    DMA 2: Programm
*/

void prog_mv(
    __SMA_RAM_PTR void* tensor_i,
    __SMA_RAM_PTR void* tensor_o,
    __SMA_RAM_PTR uint8_t* prog_base_addr,
    uint16_t prog_base_size,
    uint8_t type_size
){
    printf("Execute PROG_MV i@%p, o@%p p@%p base#%d\n", 
        tensor_i, tensor_o, prog_base_addr, prog_base_size);

    while(prog_base_size){
        // Read programm (current frame)
        printf("READ FRAME @ %p # %d\n", prog_base_addr, prog_base_size);
        DMA_INIT(2, prog_base_addr, prog_base_size);
        DMA_LD(2);
        
        struct prog_header_t *frame_header = DMA_RW(2, 0);
        printf("addr_i     = %d\n", frame_header->addr_i);
        printf("size_i     = %d\n", frame_header->size_i);
        printf("addr_o     = %d\n", frame_header->addr_o);
        printf("size_o     = %d\n", frame_header->size_o);
        printf("addr_store = %d\n", frame_header->addr_store);
        printf("size_store = %d\n", frame_header->size_store);
        printf("nb_moves   = %d\n", frame_header->nb_moves);
        printf("base_size  = %d\n", frame_header->base_size);


        // LD tensor_i if needed
        if(frame_header->addr_i != -1){
            DMA_INIT(0, (char*)tensor_i + frame_header->addr_i, frame_header->size_i);
            DMA_LD(0);
            // printf("Load DMA 0\n");
        }
        // LD tensor_o if needed
        if(frame_header->addr_o != -1){
            DMA_INIT(1, (char*)tensor_o + frame_header->addr_o, frame_header->size_o);
            DMA_LD(1);
            // printf("Load DMA 1\n");
        }
        // Process moves
        // printf("Moves #%d\n", frame_header->nb_moves);     
        for(size_t i = 0; i < frame_header->nb_moves; i++){
            int16_t addr_i = *(int16_t*)DMA_RW(2, sizeof(struct prog_header_t) + (i * 2 + 0) * sizeof(int16_t));
            // printf("addri=%d\n", addr_i);
            int16_t addr_o = *(int16_t*)DMA_RW(2, sizeof(struct prog_header_t) + (i * 2 + 1) * sizeof(int16_t));
            // printf("addro=%d\n", addr_o);
            memcpy(DMA_RW(1, addr_i), DMA_RW(0, addr_o), type_size);
        }
        // Store tensor_o if needef
        if(frame_header->addr_store != -1){
            DMA_ST(1);
            // printf("WB !!!\n");
        }

        // Update next programm frame 
        prog_base_addr += prog_base_size;
        prog_base_size = frame_header->base_size;

    }
}