
#define M 145
#define N 64

int tab0[M][N] = {0};

int main(void){
    int res = 0;

    // Write Only
    for (int j = 0; j < M; j++){
        for (int i = 0; i < N; i++){
            tab0[j][i] = i+j;
        }
    }
    
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
            res += tab0[j][i];
            res -= 1;
        }
        res -= 50;
    }

    printf("sum = %d\n", res);

    return 0;
}
