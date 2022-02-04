
#define N 64

int tab0[N][N] = {0};

int main(void){
    int res = 0;

    for (int j = 0; j < N; j++){
        res += 50;
        for (int i = 0; i < N; i++){
            res += tab0[j][i];
        }
        res -= 50;
    }

    printf("sum = %d\n", res);

    return 0;
}
