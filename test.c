


tab [C][C]; 

volatile int z = 0;

// ******************************* REFERNCE
for (int i = 0; i < C; i++){
    for int(j = 0; j < C; j++){
        tab [i][j] = i + j;
    }
}


// ******************************* inner_if
for (int i = 0; i < C; i++){
    if(i % 16 == 0){
        DMA_LOAD()
    }
    for int(j = 0; j < C; j++){
        tab [i][j] += i + j;
    }
    if(end cond){
        DMA_STORE()
    }
    
}

// ******************************* loop_division

for (int i_high = 0; i_high < 4; i_high++){
    DMA_LOAD();
    for (i_low = 0; i_low < C/4; i_low++){
        i = i_high * C/4 + i_low;
        if(i < 64){
            for int(j = 0; j < C; j++){
                tab [i][j] += i + j;
            }
        }
    }
    DMA_STORE();
}


// ******************************* loop_division + extraction residual

for (int i_high = 0; i_high < 4; i_high++){
    DMA_LOAD(DEFAULT_CONFIG);
    for (i_low = 0; i_low < C/4; i_low++){
        i = i_high * C/4 + i_low;
        for int(j = 0; j < C; j++){
            tab [i][j] += i + j;
        }
    }
    DMA_STORE(DEFAULT_CONFIG);  
}
DMA_LOAD(RESIDUAL_CONFIG);
for (i_low = 0; i_low < RESIDUAL_PART; i_low++){
    i = i_high * C/4 + i_low;
    for int(j = 0; j < C; j++){
        tab [i][j] += i + j;
    }
}
DMA_STORE(RESIDUAL_CONFIG);

// Note: Inline only the high part ? Code cost 2^N.



// *******************************
/*
^ DMA efficiency
|
|
|
|              loop_division + extraction residual
|             /                         (CPU: optimal, DMA: optimal, CODE: ---)
|            /
|           /              loop_division    (CPU: --, DMA: optimal, CODE: ++)
|          /              /
|         X              X                X---inner_if
|                                             (CPU: ---, DMA: optimal, CODE: ++)    
|
|-----------------------------------------------vvv UNoptimal DMA EFFICINCY vvv
|
|         X 
|          \ loop_division + remove residual part during optimisation
|                       (CPU: optimal, DMA +/---, CODE: optimal)
|                       DMA can be optimal, but depends on multiplicity
|
+-------------------------------------------------------------------------> CPU

*/