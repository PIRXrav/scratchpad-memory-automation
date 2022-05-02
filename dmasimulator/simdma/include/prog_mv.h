#include "dma.h"

void prog_mv(
    __SMA_RAM_PTR void* tensor_i,
    __SMA_RAM_PTR void* tensor_o,
    __SMA_RAM_PTR uint8_t* prog_base_addr,
    uint16_t prog_base_size,
    uint8_t type_size
);