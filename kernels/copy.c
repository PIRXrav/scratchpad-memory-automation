//***SMA DEF

#define N 16
#define M 2
//***SMA ARG
 
char input[N][M];
char output[N][M];


//***SMA FUN


void copy(){
    for (int n = 0; n < N; n++) {
        for(int m = 0; m < M; m++) {
            output[n][m] = input[n][m];
        }
    }
}