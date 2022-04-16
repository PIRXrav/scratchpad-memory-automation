//***SMA DEF

#define M 16
#define N 16

//***SMA ARG

/*
int32_t input[M];
int32_t output[N];
int8_t weights[N][M];
*/

int32_t input[M];
int32_t output[N];
int8_t weights[N][M];

//***SMA FUN


void gemv(){
    for (int n = 0; n < N; n++) {
        output[n] = 0;
        for (int m = 0; m < M; m++) {
            output[n] += input[m] * weights[n][m];
        }
    }
}