
#define N 64

int tab0[N][N];

int main(void){
    int res = 0;

    for (int j = 0; j < N; j++){
        for (int i = 0; i < N; i++){
            tab0[j][i] = i+j;
        }
    }

    for (int j = 0; j < N; j++){
        for (int i = 0; i < N; i++){
            res += tab0[j][i];
        }
    }

    printf("sum = %d\n", res);

    return 0;
}
