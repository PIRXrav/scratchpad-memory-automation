//***SMA DEF

#define M 16
#define N 16

//***SMA ARG
 
char input[M];
char output[N];
char weights[N][M];


//***SMA FUN


void gemv(){
    for (int n = 0; n < N; n++) {
        output[n] = 0;
        for (int m = 0; m < M; m++) {
            output[n] += input[m] * weights[n][m];
        }
    }
}