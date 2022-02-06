#include <stdio.h>

#define M 4
#define N 64

__attribute__ ((aligned (16))) char tab0[M][N] = {0};

int main(void){
    int res = 0;

    // Write Only
    for (int j = 0; j < M; j++){
        for (int i = 0; i < N; i++){
            tab0[j][i] = i+j;
        }
    }
    
    /* Debug
    for (int j = 0; j < M; j++){
        printf("j = %d :\n", j);
        for (int i = 0; i < N; i++){
            printf("%d ", tab0[j][i]);
        }
        printf("\n");
        printf("j = %d :\n", j);
    }
    */

    // Read and Write
    for (int j = 0; j < M; j++){
        for (int i = 0; i < N; i++){
            tab0[j][i] += 1;
        }
    }

    // Read Only
    for (int j = 0; j < M; j++){
        res += 50;
        for (int i = 0; i < N; i++){
            int z = i + 2;
            (void)z;
            res += tab0[j][i];
            res -= 1;
        }
        res -= 50;
    }

    printf("sum = %d\n", res);

    return 0;
}
